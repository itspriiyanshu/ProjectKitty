import firebase_admin
from firebase_admin import credentials, db
import math
import json

# === Initialize Firebase ===
cred = credentials.Certificate("../firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectkitty-9979b-default-rtdb.firebaseio.com/'
})

# === MediaPipe landmark indices ===
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# === Utility: Angle Between Vectors ===
def angle_between(a, b, c):
    ab = [a[i] - b[i] for i in range(3)]
    cb = [c[i] - b[i] for i in range(3)]
    dot = sum(ab[i] * cb[i] for i in range(3))
    norm_ab = math.sqrt(sum(ab[i] ** 2 for i in range(3)))
    norm_cb = math.sqrt(sum(cb[i] ** 2 for i in range(3)))
    if norm_ab * norm_cb == 0:
        return 0
    return math.degrees(math.acos(dot / (norm_ab * norm_cb)))

# === Extract Features from One Video ===
def process_json_data(data, label, video_id):
    features = []
    head_positions = []

    for frame in data:
        lm = frame['landmarks']
        try:
            left_elbow_angle = angle_between(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_ANKLE])
            right_elbow_angle = angle_between(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_ANKLE])
            shoulder_align = abs(lm[LEFT_SHOULDER][1] - lm[RIGHT_SHOULDER][1])
            jump_height = (lm[LEFT_ANKLE][1] + lm[RIGHT_ANKLE][1]) / 2

            hip_vec = [lm[LEFT_HIP][0] - lm[RIGHT_HIP][0], lm[LEFT_HIP][1] - lm[RIGHT_HIP][1]]
            shoulder_vec = [lm[LEFT_SHOULDER][0] - lm[RIGHT_SHOULDER][0], lm[LEFT_SHOULDER][1] - lm[RIGHT_SHOULDER][1]]
            separation_angle = angle_between(hip_vec + [0], [0, 0, 0], shoulder_vec + [0])

            head_positions.append(lm[NOSE][1])

            features.append([
                left_elbow_angle,
                right_elbow_angle,
                shoulder_align,
                jump_height,
                separation_angle
            ])
        except:
            continue

    if not features:
        return None  # Skip this sample

    # Compute head stability (variance) and insert it as feature 5 for each frame
    variance = sum((y - sum(head_positions)/len(head_positions))**2 for y in head_positions) / len(head_positions)
    for f in features:
        f.insert(5, variance)

    return {
        "video_id": video_id,
        "sequence": features,
        "label": label
    }

# === Fetch All Videos from Firebase ===
def download_and_process(folder_name, label):
    ref = db.reference(f'pose_data/{folder_name}')
    all_sequences = []
    for key, json_data in ref.get().items():
        print(f"Processing: {key}")
        sequence_dict = process_json_data(json_data, label, key)
        if sequence_dict:
            all_sequences.append(sequence_dict)
    return all_sequences

# === Entry Point ===
if __name__ == "__main__":
    good_sequences = download_and_process('goodb', 1)
    bad_sequences = download_and_process('badb', 0)

    output_file = "bowling_sequences.json"
    with open(output_file, 'w') as f:
        json.dump(good_sequences + bad_sequences, f)

    print(f"✅ LSTM-ready sequences saved to {output_file}")
