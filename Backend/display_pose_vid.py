import cv2
import mediapipe as mp
import numpy as np
import json
import argparse

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Helper to get 2D landmark coordinates
def get_coords(landmarks, idx, shape):
    lm = landmarks[idx]
    return int(lm.x * shape[1]), int(lm.y * shape[0])

# Open video
# video_path = "assets/deadpool_cut.mp4"
# cap = cv2.VideoCapture(video_path)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Display and analyze pose data from a video file')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('output_path', help='Path to save the output video file')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        exit()

    # Create a video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(args.output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    paused = False
    frame_count = 0
    all_angles = {}

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            shape = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                joints = {
                    "left_shoulder": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    "left_elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    "left_hip": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                    "right_hip": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                    "left_knee": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    "right_knee": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    "left_foot": (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                    "right_foot": (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                }

                frame_angles = {}

                for label, (a, b, c) in joints.items():
                    p1 = get_coords(lm, a, shape)
                    p2 = get_coords(lm, b, shape)
                    p3 = get_coords(lm, c, shape)

                    angle = calculate_angle(p1, p2, p3)
                    frame_angles[label] = angle

                    # Draw angle on the frame
                    cv2.putText(frame, f"{label}: {int(angle)}°", (p2[0]+5, p2[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                # Store the angles for the current frame
                all_angles[f"frame_{frame_count}"] = frame_angles
            frame_count += 1

            # Write the frame with overlay to the video
            out.write(frame)

        cv2.imshow('Pose Angles', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space to pause/resume
            paused = not paused
        elif key == ord('q'):  # Q to quit
            break

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
