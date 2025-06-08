import cv2
import mediapipe as mp
import json
import argparse
import os

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark

# Only include upper-body and leg joints
KEYPOINTS = [
    f"{side}_{bodypart}"
    for side in ["LEFT", "RIGHT"]
    for bodypart in ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
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
    parser = argparse.ArgumentParser(
        description="Analyze and optionally save pose data from a video file"
    )
    parser.add_argument(
        "video_name", help="Name of the video file (without .mp4 extension)"
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save pose data to JSON file"
    )
    parser.add_argument("-v", "--video", action="store_true", help="Save overlay video")
    parser.add_argument(
        "-d", "--display", action="store_true", help="Display video while processing"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Enable all features (save JSON, save video, and display)",
    )
    args = parser.parse_args()

    # If -a flag is used, enable all features
    if args.all:
        args.save = True
        args.video = True
        args.display = True

    # Construct input and output paths
    input_path = os.path.join(
        "..", "..", "Frontend", "public", "Videos", f"{args.video_name}.mp4"
    )
    json_output_path = os.path.join("..", "Data", f"{args.video_name}.json")
    overlay_output_path = os.path.join("..", "Videos", f"{args.video_name}_Overlay.mp4")

    # Ensure the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Video file not found at {input_path}")
        exit()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    output_data = {}

    # Create a video writer to save the output if video saving is enabled
    out = None
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            overlay_output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
        )
        print(f"Overlay video will be saved to: {overlay_output_path}")

    print(f"Processing video: {input_path}")
    if args.save:
        print(f"Pose data will be saved to: {json_output_path}")

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Extract landmarks for JSON if save flag is set
                if args.save:
                    landmarks = extract_landmarks(results)
                    if landmarks:
                        timestamp = round(frame_number / fps, 3)
                        output_data[timestamp] = landmarks

            # Write the frame with overlay to the video if video saving is enabled
            if args.video:
                out.write(frame)

            # Display the frame if display is enabled
            if args.display:
                cv2.imshow("Pose Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

    # Clean up
    cap.release()
    if args.video:
        out.release()
    if args.display:
        cv2.destroyAllWindows()

    # Save JSON data if save flag is set
    if args.save and output_data:
        with open(json_output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(output_data)} frames to {json_output_path}")

    print("Processing complete!")


if __name__ == "__main__":
    main()
