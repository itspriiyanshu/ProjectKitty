import cv2
import json
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from ultralytics import YOLO

# Load your trained YOLOv8 bowler detection model
yolo_model = YOLO("runs/detect/train5/weights/best.pt")

def visualize_keypoints_on_video(video_path, json_path, output_path="output_with_pose.mp4"):
    # Load keypoints from JSON
    with open(json_path, 'r') as f:
        keypoints_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose_connections = mp_pose.POSE_CONNECTIONS

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= len(keypoints_data):
            break

        # Run YOLO on full frame to get bowler bbox
        results = yolo_model(frame, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()

        if len(detections) == 0:
            # No detection, just write raw frame
            out.write(frame)
            idx += 1
            continue

        # Take the most confident bbox
        x1, y1, x2, y2, conf, cls = max(detections, key=lambda x: x[4])
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw bbox on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        keypoints = keypoints_data[idx]

        landmarks = []
        for (kx, ky, kz) in keypoints:
            # Convert bbox-relative coords to full frame coords
            abs_x = x1 + kx
            abs_y = y1 + ky
            abs_z = kz  # z is relative depth, keep as is or scale as needed

            # Normalize landmarks for MediaPipe drawing (values [0,1] relative to full frame)
            norm_x = abs_x / w
            norm_y = abs_y / h

            landmarks.append(landmark_pb2.NormalizedLandmark(x=norm_x, y=norm_y, z=abs_z))

        landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=landmarks)

        # Draw landmarks on frame
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmark_list,
            connections=pose_connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"[INFO] Visualization saved to {output_path}")


if __name__ == "__main__":
    video_path = "cuts/badb_vids/slow_std_video_2_runup_0.mp4"
    json_path = "cuts/badb_unprocessed_json/slow_std_video_2_runup_0.json"  # Must match number of frames
    visualize_keypoints_on_video(video_path, json_path)
