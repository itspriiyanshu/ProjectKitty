import cv2
import subprocess
import os
import time

os.makedirs("launch_vids", exist_ok=True)
os.makedirs("launch_processed", exist_ok=True)
def record_video():
    cap = cv2.VideoCapture(0)
    output_size = (1280, 720)
    out = None

    print("📷 Press 's' to start recording.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        cv2.imshow("Ready to Record", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # start recording after 's' key pressed
            break
        elif key == 27:  # ESC to exit without recording
            cap.release()
            cv2.destroyAllWindows()
            print("❌ Recording cancelled.")
            return

    # Countdown timer (3 seconds)
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        countdown_text = f"Recording starts in {i}..."
        cv2.putText(frame, countdown_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("Countdown", frame)
        cv2.waitKey(1000)  # wait 1 second

    print("🎥 Recording started. Press ESC to stop.")
    out = cv2.VideoWriter("launch_vids/raw_bowling.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, output_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_size)
        out.write(frame)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Saved raw_bowling.mp4")
    

def slow_down_video(input_path="launch_vids/raw_bowling.mp4", output_path="launch_processed/slow_bowling_30fps.mp4"):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "setpts=2.0*PTS,scale=1280:720,fps=30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Saved {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process video: {e}")


def run_prediction_pipeline():
    print("🧠 Running prediction and feedback pipeline...")
    subprocess.run(["python", "workable.py", "launch_processed/slow_bowling_30fps.mp4"])
    print("✅ Finished analysis.")

if __name__ == "__main__":
    record_video()
    slow_down_video()
    run_prediction_pipeline()
