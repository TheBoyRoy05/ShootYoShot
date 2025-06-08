from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import numpy as np
import copy

# ======== Data Models ========

# Represents a single pose frame with landmarks (joint positions)
class PoseType(BaseModel):
    frame: Optional[int] = None  # Optional frame index
    timestamp: float  # Time of this pose frame
    landmarks: Dict[str, List[float]]  # Landmark coordinates (x, y, z) for each joint

# Incoming request with actual and expected pose sequences
class ScoreRequest(BaseModel):
    actual: List[PoseType]
    expected: List[PoseType]

# ======== FastAPI Setup ========

app = FastAPI()

# Allow cross-origin requests from frontend (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint to confirm server is running
@app.get("/health")
def health():
    return {"status": "ok"}

# Score endpoint to compare actual vs. expected movement
@app.post("/score")
def score(request: ScoreRequest):
    actual = request.actual
    expected = request.expected
    print(len(actual), len(expected))  # Debug output

    # Compute similarity score, then scale and clamp it between 0 and 100
    score = calculate_highest_grade(expected, actual) * 1000
    score = (score - 75) * 10 + 75
    return {"score": min(max(score, 0), 100)}

# ======== Pose Evaluation Logic ========

# Joint pairs used to form vectors (e.g. elbow to wrist)
REL_VEC_TUPS = (
    ("LEFT_WRIST", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_SHOULDER"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("RIGHT_WRIST", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
)

# Interpolate pose at a given timestamp between two frames
def lin_interpolate_frames(frame1, frame2, timestamp):
    output_landmarks = {}
    output = PoseType(frame=None, timestamp=timestamp, landmarks=output_landmarks)

    # List of landmark keys to interpolate
    landmarks = [
        f"{side}_{bodypart}"
        for side in ["LEFT", "RIGHT"]
        for bodypart in ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
    ]
    
    # Time ratio for interpolation
    fraction = (timestamp - frame1.timestamp) / (frame2.timestamp - frame1.timestamp)

    for landmark in landmarks:
        # Interpolate each coordinate (x, y, z)
        output_landmarks[landmark] = [
            frame1.landmarks[landmark][dim] +
            fraction * (frame2.landmarks[landmark][dim] - frame1.landmarks[landmark][dim])
            for dim in range(3)
        ]

    return output

# Normalize and align timestamps between two pose sequences using linear interpolation
def fill_values(template_pose_data, user_pose_data):
    all_timestamps_ordered = [-1]
    num_points_temp = len(template_pose_data)
    num_points_user = len(user_pose_data)
    temp_pointer = 0
    user_pointer = 0
    temp_orig_timestamps = set()
    user_orig_timestamps = set()

    # Merge timestamps from both sources into a single sorted list
    while True:
        try:
            curr_temp = template_pose_data[temp_pointer].timestamp
            curr_user = user_pose_data[user_pointer].timestamp
        except:
            break

        if temp_pointer >= num_points_temp and all_timestamps_ordered[-1]:
            if user_pointer >= num_points_user:
                break
            if all_timestamps_ordered[-1] != curr_user:
                all_timestamps_ordered.append(curr_user)
                user_orig_timestamps.add(curr_user)
            user_pointer += 1
        elif user_pointer >= num_points_user:
            if all_timestamps_ordered[-1] != curr_temp:
                all_timestamps_ordered.append(curr_temp)
                temp_orig_timestamps.add(curr_temp)
            temp_pointer += 1
        elif curr_temp < curr_user:
            if all_timestamps_ordered[-1] != curr_temp:
                all_timestamps_ordered.append(curr_temp)
                temp_orig_timestamps.add(curr_temp)
            temp_pointer += 1
        else:
            if all_timestamps_ordered[-1] != curr_user:
                all_timestamps_ordered.append(curr_user)
                user_orig_timestamps.add(curr_user)
            user_pointer += 1

    all_timestamps_ordered = all_timestamps_ordered[1:]

    temp_out = []
    user_out = []

    temp_new_pointer = 0
    user_new_pointer = 0

    for timestamp in all_timestamps_ordered:
        # Append real or interpolated frames for template
        if timestamp in temp_orig_timestamps:
            temp_out.append(template_pose_data[temp_new_pointer])
            temp_new_pointer += 1
        else:
            temp_out.append(
                lin_interpolate_frames(
                    template_pose_data[temp_new_pointer - 1],
                    template_pose_data[temp_new_pointer],
                    timestamp,
                )
            )
        # Append real or interpolated frames for user
        if timestamp in user_orig_timestamps:
            user_out.append(user_pose_data[user_new_pointer])
            user_new_pointer += 1
        else:
            user_out.append(
                lin_interpolate_frames(
                    user_pose_data[user_new_pointer - 1],
                    user_pose_data[user_new_pointer],
                    timestamp,
                )
            )

    return temp_out, user_out

# Compute unit vector from one joint to another
def find_normalized_relative_vec_from_obj(from_landmark, to_landmark, frame_obj):
    from_coords = frame_obj.landmarks[from_landmark]
    to_coords = frame_obj.landmarks[to_landmark]
    relative_vec = [to_coords[i] - from_coords[i] for i in range(3)]
    norm = np.linalg.norm(relative_vec)
    return [c / norm for c in relative_vec]

# Assign weight to each joint vector based on how much it changes over time
def find_weights(pose_data):
    weights = []
    for rel_vec in REL_VEC_TUPS:
        diffs = []
        for i in range(1, len(pose_data)):
            vec1 = find_normalized_relative_vec_from_obj(*rel_vec, pose_data[i - 1])
            vec2 = find_normalized_relative_vec_from_obj(*rel_vec, pose_data[i])
            diffs.append(np.linalg.norm(np.array(vec2) - np.array(vec1)))
        weights.append(sum(diffs))
    return weights

# Calculate expected vs actual vectors between joints at one frame
def calculate_vectors(expected, actual):
    exp_and_actual_vec = {}
    for key in REL_VEC_TUPS:
        expected_vector = np.array(expected[key[0]]) - np.array(expected[key[1]])
        actual_vector = np.array(actual[key[0]]) - np.array(actual[key[1]])
        exp_and_actual_vec[key] = [(expected_vector, actual_vector)]
    return exp_and_actual_vec

# Score the similarity between two pose sequences based on vector alignment
def calculate_norm(expected_frames, actual_frames, weights):
    total = np.array([])
    for i in range(len(expected_frames)):
        vectors = calculate_vectors(expected_frames[i], actual_frames[i])
        list_of_diffs = np.array([])

        for num, key in enumerate(REL_VEC_TUPS):
            vector1, vector2 = vectors[key][0]
            cosine_similarity = np.dot(vector1, vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
            list_of_diffs = np.append(
                list_of_diffs, (cosine_similarity + 1) / 2 * weights[num]
            )

        avg = list_of_diffs.sum() / len(list_of_diffs)
        total = np.append(total, avg)

    return total.mean()

# Softmax to normalize weights
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Stabilize values for large exponentials
    return e_x / e_x.sum(axis=0, keepdims=True)

# Try matching segments of actual movements to expected movement and grade them
def calculate_grade_for_groups2(expected_movements, actual_movements):
    weights = softmax(find_weights(expected_movements))
    grade_per_timestamp_group = {}

    for i in range(len(expected_movements), len(actual_movements)):
        rang = (i - len(expected_movements), i)
        actual_movements_window = copy.deepcopy(actual_movements[rang[0]: rang[1]])
        first_time = actual_movements_window[0].timestamp

        # Normalize timestamps to start from zero
        for j in range(len(actual_movements_window)):
            actual_movements_window[j].timestamp -= first_time

        interpolated_data = fill_values(expected_movements, actual_movements_window)
        expected_landmarks = [d.landmarks for d in interpolated_data[0]]
        actual_landmarks = [d.landmarks for d in interpolated_data[1]]
        grade = calculate_norm(expected_landmarks, actual_landmarks, weights)
        grade_per_timestamp_group[rang] = grade

    return grade_per_timestamp_group

# Get the highest scoring match from all possible subsequences
def calculate_highest_grade(expected_movements, actual_movements):
    return max(calculate_grade_for_groups2(expected_movements, actual_movements).values())
