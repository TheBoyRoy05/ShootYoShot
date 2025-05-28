import cv2
import mediapipe as mp
import json
import argparse
import os

# Remove hardcoded paths
FRAME_INTERVAL = 1  # Process every frame

mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark

# Only include upper-body and leg joints
KEYPOINTS = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    # "NOSE",
]

def extract_landmarks(results):
    landmark_dict = {}
    if not results.pose_world_landmarks:
        return None

    for key in KEYPOINTS:
        idx = getattr(POSE_LANDMARKS, key).value
        lm = results.pose_world_landmarks.landmark[idx]
        landmark_dict[key] = [lm.x, -lm.y, -lm.z]

    return landmark_dict

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract pose data from a video file')
    parser.add_argument('video_name', help='Name of the video file (without .mp4 extension)')
    args = parser.parse_args()

    # Construct input and output paths
    input_path = os.path.join('..', 'Data', 'Videos', f"{args.video_name}.mp4")
    output_path = os.path.join('..', 'Data', f"{args.video_name}.json")

    # Ensure the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        exit()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    output = []

    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % FRAME_INTERVAL == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                landmarks = extract_landmarks(results)
                if landmarks:
                    output.append(
                        {
                            "frame": frame_number,
                            "timestamp": round(frame_number / fps, 3),
                            "landmarks": landmarks,
                        }
                    )
            frame_number += 1
    cap.release()

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(output)} frames to {output_path}")

if __name__ == "__main__":
    main()
