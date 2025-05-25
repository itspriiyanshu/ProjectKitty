import cv2
import mediapipe as mp
import numpy as np
import os
from moviepy.editor import VideoFileClip

def detect_runup_segments(video_path, output_dir, fps_expected=30):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = fps_expected
        print(f"Warning: Could not detect FPS, defaulting to {fps_expected}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    movement = []
    tracking = False
    start_frame = None
    last_end_frame = -1

    def extract_points(landmarks, indices):
        if landmarks is None:
            return None
        visible = lambda lm: lm.visibility > 0.5
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            if visible(lm):
                pts.append((lm.x, lm.y))
            else:
                pts.append(None)
        pts_filtered = [p for p in pts if p is not None]
        return pts_filtered if pts_filtered else None

    def significant_movement(curr_points, prev_points, threshold=0.02, require_downward=False):
        if prev_points is None or curr_points is None:
            return False
        for (x1, y1), (x2, y2) in zip(curr_points, prev_points):
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dy = y2 - y1  # y increases downward in image coordinates
            if dist > threshold:
                if require_downward and dy <= 0:
                    continue
                return True
        return False

    prev_ankle_points = None
    prev_shoulder_points = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        curr_ankle_points = extract_points(landmarks, [27, 28])
        curr_shoulder_points = extract_points(landmarks, [11, 12])

        ankle_moved = significant_movement(curr_ankle_points, prev_ankle_points, require_downward=True)
        shoulder_moved = significant_movement(curr_shoulder_points, prev_shoulder_points)

        prev_ankle_points = curr_ankle_points
        prev_shoulder_points = curr_shoulder_points

        moved = ankle_moved and shoulder_moved

        if moved:
            if not tracking:
                start_frame = max(frame_idx - int(1 * fps), last_end_frame + 1)  # ensure no overlap
                tracking = True
        elif tracking:
            end_frame = min(frame_idx + int(1 * fps), frame_count - 1)
            movement.append((start_frame, end_frame))
            last_end_frame = end_frame
            tracking = False

        frame_idx += 1

    if tracking:
        end_frame = min(frame_count - 1, frame_idx + int(1 * fps))
        movement.append((start_frame, end_frame))

    cap.release()
    pose.close()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with VideoFileClip(video_path) as clip:
        for i, (start_f, end_f) in enumerate(movement):
            duration = end_f - start_f
            if duration < 15:
                print(f"Skipping short segment: frames {start_f}-{end_f}, duration {duration}")
                continue

            start_t = start_f / fps
            end_t = end_f / fps
            out_name = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_runup_{i}.mp4")
            print(f"Saving clip {i} from {start_t:.2f}s to {end_t:.2f}s")
            subclip = clip.subclip(start_t, end_t)
            subclip.write_videofile(out_name, codec="libx264", audio=False)

def run_on_folder(video_dir, output_dir):
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(videos)} videos in {video_dir}")
    for video in videos:
        video_path = os.path.join(video_dir, video)
        detect_runup_segments(video_path, output_dir)

if __name__ == "__main__":
    input_videos_dir = "slowed_videos"       # folder containing standardized slowed videos
    output_clips_dir = "output"               # folder to store runup clips
    run_on_folder(input_videos_dir, output_clips_dir)
