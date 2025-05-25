import json
import os

fps = 30  # your standardized frame rate

def process_pose_json(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)  # data is a list of frames: [[[x,y,z], ...], ...]

    processed_frames = []
    for i, frame_keypoints in enumerate(data):
        frame_dict = {
            "frame_index": i,
            "timestamp": round(i / fps, 5),
            "landmarks": frame_keypoints
        }
        processed_frames.append(frame_dict)

    with open(output_path, 'w') as f:
        json.dump(processed_frames, f, indent=2)
    print(f"Processed {os.path.basename(input_path)} -> {output_path}")

def batch_process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            input_path = os.path.join(folder_path, file)
            output_name = os.path.splitext(file)[0] + '_processed.json'
            output_path = os.path.join(folder_path, output_name)
            process_pose_json(input_path, output_path)

if __name__ == "__main__":
    folders = ['cuts/goodb', 'cuts/badb']
    for folder in folders:
        print(f"Processing folder: {folder}")
        batch_process_folder(folder)
    print("All JSON files processed!")
