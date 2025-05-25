import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
import json
import os
import gc
from tqdm import tqdm
from ultralytics import YOLO
from lstm_train import Attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
from custom_layer import Attention

MODEL_PATH = "bowling_lstm_classifier.keras"
BASELINE_PATH = "baseline_good_curve.json"  # Store average curves of good bowling actions
POSE_LANDMARKS = {
    'NOSE': 0, 'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
}
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# @register_keras_serializable()
# class Attention(Layer):
#     def __init__(self, **kwargs):
#         super(Attention, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
#                                  initializer="normal")
#         self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
#                                  initializer="zeros")
#         super(Attention, self).build(input_shape)

#     def call(self, inputs):
#         e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
#         a = tf.keras.backend.softmax(e, axis=1)
#         output = inputs * a
#         return tf.keras.backend.sum(output, axis=1)
model = tf.keras.models.load_model(MODEL_PATH,custom_objects={"Attention": Attention})

with open(BASELINE_PATH) as f:
    BASELINE_CURVES = json.load(f)

def angle_between(a, b, c):
    ab = [a[i] - b[i] for i in range(3)]
    cb = [c[i] - b[i] for i in range(3)]
    dot = sum(ab[i] * cb[i] for i in range(3))
    norm_ab = math.sqrt(sum(ab[i]**2 for i in range(3)))
    norm_cb = math.sqrt(sum(cb[i]**2 for i in range(3)))
    return math.degrees(math.acos(dot / (norm_ab * norm_cb))) if norm_ab * norm_cb != 0 else 0

def extract_features_from_landmarks(sequence):
    features = []
    head_positions = []

    for lm in sequence:
        try:
            left_elbow_angle = angle_between(lm[POSE_LANDMARKS['LEFT_SHOULDER']], lm[POSE_LANDMARKS['LEFT_ELBOW']], lm[POSE_LANDMARKS['LEFT_ANKLE']])
            right_elbow_angle = angle_between(lm[POSE_LANDMARKS['RIGHT_SHOULDER']], lm[POSE_LANDMARKS['RIGHT_ELBOW']], lm[POSE_LANDMARKS['RIGHT_ANKLE']])
            shoulder_align = abs(lm[POSE_LANDMARKS['LEFT_SHOULDER']][1] - lm[POSE_LANDMARKS['RIGHT_SHOULDER']][1])
            jump_height = (lm[POSE_LANDMARKS['LEFT_ANKLE']][1] + lm[POSE_LANDMARKS['RIGHT_ANKLE']][1]) / 2
            hip_vector = np.subtract(lm[POSE_LANDMARKS['LEFT_HIP']], lm[POSE_LANDMARKS['RIGHT_HIP']])
            shoulder_vector = np.subtract(lm[POSE_LANDMARKS['LEFT_SHOULDER']], lm[POSE_LANDMARKS['RIGHT_SHOULDER']])
            separation_angle = angle_between(list(hip_vector) + [0], [0,0,0], list(shoulder_vector) + [0])
            head_positions.append(lm[POSE_LANDMARKS['NOSE']][1])

            features.append([left_elbow_angle, right_elbow_angle, shoulder_align, jump_height, separation_angle])
        except:
            continue

    if head_positions:
        head_variance = np.var(head_positions)
        for f in features:
            f.insert(5, head_variance)

    return features

yolo_model = YOLO("runs/detect/train5/weights/best.pt")

def extract_pose_sequence_from_video(video_path):
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

# ========== Feedback Generation ==========
def generate_feedback(features, baseline_curves, tolerance=10.0):
    features = np.array(features).T  # (features, timesteps)
    baseline = np.array([baseline_curves[k] for k in [
        "left_elbow_angle", "right_elbow_angle", "shoulder_alignment",
        "jump_height", "head_stability", "hip_shoulder_separation"
    ]])

    feedback = []
    feature_names = [
        "Left Elbow Angle", "Right Elbow Angle", "Shoulder Alignment",
        "Jump Height", "Head Stability", "Hip-Shoulder Separation"
    ]
    
    detailed_feedback = {}

    for i in range(features.shape[0]):
        min_len = min(len(features[i]), len(baseline[i]))
        # Signed mean deviation (feature - baseline)
        signed_deviation = features[i][:min_len] - baseline[i][:min_len]
        abs_deviation = abs(signed_deviation)

        if np.mean(abs_deviation) > tolerance:
            timestamps = list(np.where(abs_deviation > tolerance)[0])
            direction = "above" if np.mean(signed_deviation) > 0 else "below"
            correction = ""

            if feature_names[i] == "Left Elbow Angle" or feature_names[i] == "Right Elbow Angle":
                if direction == "above":
                    correction = "Try to relax your elbow more during the run-up."
                else:
                    correction = "Try to bend your elbow more to improve your bowling form."

            elif feature_names[i] == "Shoulder Alignment":
                if direction == "above":
                    correction = "Your shoulders may be too rotated; focus on aligning them squarely."
                else:
                    correction = "You might be under-rotating your shoulders; increase rotation for power."

            elif feature_names[i] == "Jump Height":
                if direction == "above":
                    correction = "You are jumping too high; control your jump for better balance."
                else:
                    correction = "Try to increase your jump height for more momentum."

            elif feature_names[i] == "Head Stability":
                if direction == "above":
                    correction = "Your head movement is excessive; keep your head steady during delivery."
                else:
                    correction = "You might be too rigid; allow slight natural head movement."

            elif feature_names[i] == "Hip-Shoulder Separation":
                if direction == "above":
                    correction = "Your hip-shoulder separation is excessive; try to sync your hips and shoulders better."
                else:
                    correction = "Increase hip-shoulder separation to generate more torque."

            feedback.append(
                # f"⚠️ {feature_names[i]} is {np.mean(abs_deviation):.2f} units {direction} ideal. {correction}"
                f"⚠️ {feature_names[i]} is {direction} ideal. {correction}"
            )
            detailed_feedback[feature_names[i]] = {
                "deviation": np.mean(abs_deviation),
                "sign": direction,
                "timestamps": timestamps,
                "correction": correction
            }

    return feedback, detailed_feedback or ["✅ Good form across all key features."]

# import cv2
# import json

# import cv2
# import json
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
from mediapipe.framework.formats import landmark_pb2




def visualize_keypoints_on_video(video_path, json_path, output_path="detected_with_pose.mp4"):
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

    

# ========== Main Pipeline ==========
def analyze_bowling_video(video_path):
    print(f"\n🎥 Analyzing video: {video_path}")
    # sequence = extract_pose_sequence_from_video(video_path)
    sequence = extract_pose_sequence_from_video(video_path)
    with open('captured_video.json', 'w') as f:
        json.dump(sequence, f)
    
    with open('captured_video.json', 'r') as f:
        data = json.load(f)

    processed_frames = []
    for i, frame_keypoints in enumerate(data):
        frame_dict = {
            "frame_index": i,
            "timestamp": round(i / 30, 5),
            "landmarks": frame_keypoints
        }
        processed_frames.append(frame_dict)

    with open('captured_video_processed.json', 'w') as f:
        json.dump(processed_frames, f, indent=2)
    
    features = extract_features_from_landmarks(sequence)
    
    if not features or len(features[0]) != 6:
        print("❌ Error: Could not extract valid features from the video. Check the quality and ensure the bowler is visible.")
        return

    X = pad_sequences([features], maxlen=287, dtype='float32', padding='post')  # adjust maxlen as needed

    prediction = model.predict(X)[0][0]
    
    threshold = 0.50  # set your desired threshold here
    label = 1 if prediction >= threshold else 0

    print(f"📊 Predicted Label: {'Good' if label == 1 else 'Bad'} (Prediction: {prediction:.2f})")
    visualize_keypoints_on_video(video_path, 'captured_video.json')

    if label == 0:
        feedback, detailed_feedback = generate_feedback(features, BASELINE_CURVES)
        print("\n🧠 Corrective Feedback:")
        for f in feedback:
            print(f)
    else:
        print("\n✅ Bowling action looks good!")

# ========== Entry ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bowling_analyzer.py path_to_video.mp4")
    else:
        analyze_bowling_video(sys.argv[1])
