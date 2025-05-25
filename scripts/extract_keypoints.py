import os
import cv2
import json
import gc
import mediapipe as mp
from tqdm import tqdm
from ultralytics import YOLO

# Load your trained YOLOv8 bowler detection model
yolo_model = YOLO("runs/detect/train5/weights/best.pt")

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO to detect the bowler
        results = yolo_model(frame, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()

        if len(detections) == 0:
            # No bowler found, skip frame or fill with zero keypoints
            keypoints_sequence.append([(0.0, 0.0, 0.0)] * 33)
            continue

        # Use the most confident detection
        x1, y1, x2, y2, conf, cls = max(detections, key=lambda x: x[4])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop the bowler region
        cropped = frame[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Run MediaPipe Pose on cropped image
        results = mp_pose.process(cropped_rgb)

        if results.pose_landmarks:
            frame_kps = []
            for lm in results.pose_landmarks.landmark:
                # Normalize keypoints to cropped image dimensions
                cx = lm.x * (x2 - x1)
                cy = lm.y * (y2 - y1)
                cz = lm.z * (x2 - x1)  # assume width for depth scaling
                frame_kps.append((cx, cy, cz))
        else:
            frame_kps = [(0.0, 0.0, 0.0)] * 33

        keypoints_sequence.append(frame_kps)

    cap.release()
    mp_pose.close()
    gc.collect()
    return keypoints_sequence

def process_all_videos(input_dir, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    videos = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    for video_file in tqdm(videos, desc=f"Processing {label}"):
        try:
            video_path = os.path.join(input_dir, video_file)
            json_path = os.path.join(output_dir, video_file.replace('.mp4', '.json'))

            if os.path.exists(json_path):
                continue

            keypoints = extract_keypoints_from_video(video_path)

            with open(json_path, 'w') as f:
                json.dump(keypoints, f)

        except Exception as e:
            print(f"[ERROR] Failed on {video_file}: {e}")
            continue

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    cuts_path = os.path.join(base_path, 'cuts')

    process_all_videos(
        input_dir=os.path.join(cuts_path, 'goodb_vids'),
        output_dir=os.path.join(cuts_path, 'goodb'),
        label="Good Deliveries"
    )

    process_all_videos(
        input_dir=os.path.join(cuts_path, 'badb_vids'),
        output_dir=os.path.join(cuts_path, 'badb'),
        label="Bad Deliveries"
    )
