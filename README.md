# ProjectKitty
---

## 🔧 Setup Instructions

Before running any script, create a virtual environment and install the required packages.

### 1. Create and activate the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
---

## 🧠 Script Descriptions

| File                            | Description                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `baseline_curve.py`             | Defines baseline motion curves (e.g., joint angles) for comparison during corrective feedback.               |
| `baseline_good_curve.json`      | Stores baseline feature curves of ideal bowling actions. Used for comparison.                                |
| `bowling_lstm_classifier.keras` | Trained LSTM model for classifying good/bad bowling actions.                                                 |
| `check_keypoints.py`            | Validates extracted keypoints from pose estimation; checks for completeness or anomalies.                    |
| `convert_json.py`               | Converts raw JSON pose data to the internal standardized format used by the analysis pipeline.               |
| `custom_layer.py`               | Contains custom Keras layers or attention modules used in the LSTM model.                                    |
| `detect_runups.py`              | Identifies bowling run-up segments in slowed/stabilized videos using pose or movement heuristics.            |
| `extract_features.py`           | Converts keypoints to feature vectors such as joint angles, velocities, or body alignment metrics.           |
| `extract_keypoints.py`          | Runs MediaPipe Pose (optionally after YOLO bowler detection) and outputs structured keypoints in JSON.       |
| `launch.py`                     | Unified script to run the complete pipeline: from video input → pose → features → classification → feedback. |
| `lstm_train.py`                 | Trains the LSTM model using the extracted pose features and labeled sequences.                               |
| `m3u8_downloader.py`            | Downloads cricket videos (e.g., match highlights) from `.m3u8` links. Useful for dataset creation.           |
| `slowmo_standardise.py`         | Slows down and stabilizes cricket videos to standard frame rate for consistent pose detection.               |
| `workable.py`                   | Full pipeline script to analyse input videos and give predictions with corrective feedback.                         |
| `firebase_pose_features2.csv`   | Feature data collected from Firebase or live upload for training or analysis.                                |
| `bowling_pose_dataset.csv`      | Master dataset containing pose features and labels for LSTM training.                                        |
| `captured_video.json`           | Raw keypoints data from a recorded video.                                                                    |
| `captured_video_processed.json` | Cleaned and standardized version of the above, ready for model input.                                        |
| `bowling_sequences.json`        | Bowling video segments with annotations (start/end frames).                                                  |
| `good_bowling_sequences.json`   | Subset of sequences labeled as good bowling actions.                                                         |
| `training_history.json`         | Stores model training metrics such as accuracy and loss per epoch.                                           |
| `roboflow.ipynb`                | Jupyter notebook used for preparing object detection datasets via Roboflow (for YOLO).                |

---

## 📁 Folder Summary

| Folder              | Description                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `runs/`             | Likely used for storing YOLOv8/YOLOv11 inference results.                                  |
| `slowed_videos/`    | Contains slowed and stabilized versions of bowling videos.                                 |
| `output/`           | Final analysis outputs such as classification results, visualizations, or feedback videos. |
| `launch_vids/`      | Raw videos captured via the `launch.py` script.                                      |
| `launch_processed/` | Processed video clips generated after trimming and standardisation, these will be sent as input to workable.py.                         |
| `data/`             | Raw data that we started with , this was later processed and trimmed and used to train the model.                     |

---

## 🎥 Pretrained Models

| File         | Description                                                                           |
| ------------ | ------------------------------------------------------------------------------------- |
| `yolov8n.pt` | Lightweight YOLOv8 model for real-time person/bowler detection.                       |
| `yolo11n.pt` | Custom YOLO model possibly trained for better bowler segmentation or arrow detection. |

---

## 📹 Sample Video

| File                     | Description                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `detected_with_pose.mp4` | Sample video output with overlaid pose keypoints from MediaPipe. Useful for validation. |

---

## 🧪 Workflow Summary

#Real-Time

1. Run launch.py
2. It will capture your video (s to start, 3 sec timer before start, esc to end) please try to capture only the runup part.
3. `launch.py` will automatically standardise your raw input and call `workable.py` for analyis and feedback with prediction.
4. If bad action is detected, compare against `baseline_good_curve.json` for feedback.

#Mp4s

1. Run `workable.py` with a processed runup clip as sys.argv[1].
2. You can use the sample clips or process your own mp4 for input before using it. (Trimmed raw bowling clip -> `slowmo_standardise.py` -> Ready for `workable.py`)
---
