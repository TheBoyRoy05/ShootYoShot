from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Tuple
import numpy as np
import copy

LandmarkType = Dict[str, List[float]]


# Define the PoseType
class PoseType(BaseModel):
    frame: Optional[int] = None
    timestamp: float
    landmarks: LandmarkType


# Define the request body structure
class ScoreRequest(BaseModel):
    data: List[PoseType]


app = FastAPI()

# Add CORS middleware to allow your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow requests from your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(request: ScoreRequest):
    data = request.data
    print(data)
    return {"score": min(max(score, 0), 100)}


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

landmarks = [
    f"{side}_{bodypart}"
    for side in ["LEFT", "RIGHT"]
    for bodypart in ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
]

lerp = lambda a, b, t: a + t * (b - a)
normalize = lambda a, b: (b - a) / np.linalg.norm(b - a)

cosine_similarity = (
    lambda a, b: (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) + 1) / 2
)


def lerp_frames(pose1: PoseType, pose2: PoseType, timestamp: float) -> PoseType:
    frac = (timestamp - pose1.timestamp) / (pose2.timestamp - pose1.timestamp)

    return PoseType(
        frame=None,
        timestamp=timestamp,
        landmarks={
            landmark: [
                lerp(pose1.landmarks[landmark][i], pose2.landmarks[landmark][i], frac)
                for i in range(3)
            ]
            for landmark in landmarks
        },
    )


def fill_values(
    data1: List[PoseType], data2: List[PoseType]
) -> Tuple[List[PoseType], List[PoseType]]:
    """
    Main function you want to use to interpolate a bunch at once
    Takes in two pose data lists and creates new values so that both are defined on same timestamps
    """

    timestamps1 = [pose.timestamp for pose in data1]
    timestamps2 = [pose.timestamp for pose in data2]
    all_timestamps_ordered = sorted(list(set(timestamps1 + timestamps2)))

    out1, out2 = [], []
    P1, P2 = 0, 0

    for timestamp in all_timestamps_ordered:
        if timestamp in timestamps1:
            P1 += 1
            out1.append(data1[P1])
        else:
            out1.append(lerp_frames(data1[P1 - 1], data1[P1], timestamp))

        if timestamp in timestamps2:
            P2 += 1
            out2.append(data2[P2])
        else:
            out2.append(lerp_frames(data2[P2 - 1], data2[P2], timestamp))

    return out1, out2


def find_weights(poses: List[PoseType]) -> np.ndarray:
    def diff(joint1, joint2, i):
        return np.linalg.norm(
            normalize(poses[i].landmarks[joint1], poses[i].landmarks[joint2])
            - normalize(poses[i - 1].landmarks[joint1], poses[i - 1].landmarks[joint2])
        )

    return [
        sum([diff(rel_vec[0], rel_vec[1], i) for i in range(1, len(poses))])
        for rel_vec in REL_VEC_TUPS
    ]


def calculate_similarities(frame1: LandmarkType, frame2: LandmarkType) -> List[float]:
    return [
        cosine_similarity(
            np.array(frame1[key[0]]) - np.array(frame1[key[1]]),
            np.array(frame2[key[0]]) - np.array(frame2[key[1]]),
        )
        for key in REL_VEC_TUPS
    ]


def calculate_norm(
    landmarks1: List[LandmarkType], landmarks2: List[LandmarkType], weights: List[float]
) -> float:
    avgs = [
        np.dot(calculate_similarities(landmarks1[i], landmarks2[i]), weights)
        for i in range(len(landmarks1))
    ]
    return np.mean(avgs)


def softmax(x: List[float]) -> List[float]:
    e_x = np.exp(x - np.max(x))  # Shift values for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)


def calculate_grade(expected: List[PoseType], actual: List[PoseType]) -> float:
    grades = []
    weights = softmax(find_weights(expected))

    for i in range(len(expected), len(actual)):
        actual_window = copy.deepcopy(actual[i - len(expected) : i])
        first_time = actual_window[0].timestamp

        for frame in actual_window:
            frame.timestamp = frame.timestamp - first_time

        full_expected, full_actual = fill_values(expected, actual_window)
        expected_landmarks = [data.landmarks for data in full_expected]
        actual_landmarks = [data.landmarks for data in full_actual]
        
        grade = calculate_norm(expected_landmarks, actual_landmarks, weights)
        grades.append(grade)

    return max(grades)
