import os
import subprocess
import sys

# Config
output_dir = "launch_processed"
slow_factor = 2.0  # 2x slower → 50% speed
target_resolution = "1280x720"
target_fps = 30

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_video(input_path, output_path):
    """
    Slow down and standardize a video using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"setpts={slow_factor}*PTS,scale={target_resolution},fps={target_fps}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # remove audio
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Processed: {os.path.basename(output_path)}")
    except subprocess.CalledProcessError:
        print(f"[x] Failed to process: {os.path.basename(input_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_and_run.py path/to/input_video.mp4")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"[x] File does not exist: {input_path}")
        sys.exit(1)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"slow_std_{filename}")

    process_video(input_path, output_path)

# import os
# import subprocess
# import sys

# # Config
# input_dir = "data"
# output_dir = "slowed_videos"
# slow_factor = 2.0  # 2x slower → 50% speed
# target_resolution = "1280x720"
# target_fps = 30

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# def process_video(input_path, output_path):
#     """
#     Slow down and standardize a video using ffmpeg.
#     """
#     cmd = [
#         "ffmpeg",
#         "-i", input_path,
#         "-vf", f"setpts={slow_factor}*PTS,scale={target_resolution},fps={target_fps}",
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         "-an",  # remove audio
#         output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True)
#         print(f"[✓] Processed: {os.path.basename(output_path)}")
#     except subprocess.CalledProcessError:
#         print(f"[x] Failed to process: {os.path.basename(input_path)}")

# # Process all .mp4 videos in the input directory
# videos = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

# if not videos:
#     print("No videos found in data/ folder.")
# else:
#     for vid in videos:
#         input_path = os.path.join(input_dir, vid)
#         output_path = os.path.join(output_dir, f"slow_std_{vid}")
#         process_video(input_path, output_path)

# if __name__ == "__main__":
#     import sys
#     process_video(sys.argv[1])