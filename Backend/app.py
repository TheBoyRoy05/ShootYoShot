from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Tuple
import numpy as np
import json
import os

Pose = Dict[str, List[float]]
Move = Dict[float, Pose]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Optimized scoring code ---
PLAYERS = [
    "Anthony",
    "DeAndre",
    "Giannis",
    "Jeremy",
    "Kobe",
    "LeBron",
    "Rudy",
    "Shai",
    "Shaq",
    "Steph",
]

BONES = (
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

LANDMARKS = [
    f"{side}_{bodypart}"
    for side in ["LEFT", "RIGHT"]
    for bodypart in ["WRIST", "ELBOW", "SHOULDER", "HIP", "KNEE", "ANKLE"]
]

BONE_INDICES = [(LANDMARKS.index(j1), LANDMARKS.index(j2)) for j1, j2 in BONES]


def lerp(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return x + t * (y - x)


def lerp_pose(pose1: Pose, pose2: Pose, t: float) -> Pose:
    return {
        lm: lerp(np.array(pose1[lm]), np.array(pose2[lm]), t).tolist()
        for lm in LANDMARKS
    }


def fill_values(move1: Move, move2: Move) -> Tuple[Move, Move]:
    times1 = np.array(list(move1.keys())) - move1.keys()[0]
    times2 = np.array(list(move2.keys())) - move2.keys()[0]
    all_times = np.sort(np.unique(np.concatenate([times1, times2])))

    def fill(move: Move, times: np.ndarray) -> Move:
        poses = list(move.values())
        out = {}

        for time in all_times:
            if time in times:
                idx = np.where(times == time)[0][0]
                out[time] = poses[idx]
            else:
                idx = np.searchsorted(times, time)
                if idx == 0:
                    out[time] = poses[0]
                elif idx == len(times):
                    out[time] = poses[-1]
                else:
                    t1, t2 = times[idx - 1], times[idx]
                    frac = (time - t1) / (t2 - t1)
                    out[time] = lerp_pose(poses[idx - 1], poses[idx], frac)
        return out

    return fill(move1, times1), fill(move2, times2)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float((np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) + 1) / 2)


def calculate_similarities(pose1: Pose, pose2: Pose) -> np.ndarray:
    pose1_arr = np.array([pose1[lm] for lm in LANDMARKS])
    pose2_arr = np.array([pose2[lm] for lm in LANDMARKS])

    similarities = np.zeros(len(BONES))
    for i, (idx1, idx2) in enumerate(BONE_INDICES):
        vec1 = pose1_arr[idx2] - pose1_arr[idx1]
        vec2 = pose2_arr[idx2] - pose2_arr[idx1]
        similarities[i] = cosine_similarity(vec1, vec2)
    return similarities


def similarity_score(move1: Move, move2: Move) -> float:
    move1_vals = list(move1.values())
    move2_vals = list(move2.values())

    similarities = np.array(
        [
            calculate_similarities(move1_vals[i], move2_vals[i])
            for i in range(len(move1))
        ]
    )
    return float(np.mean(similarities))


def rescale(x: float, lower: float, upper: float) -> float:
    return float(np.clip(1 - ((x - upper) / (lower - upper)), 0, 1))


def sliding_window_score(move1: Move, move2: Move) -> List[float]:
    if len(move1) > len(move2):
        move1, move2 = move2, move1

    move2_items = list(move2.items())
    scores = np.zeros(len(move2) - len(move1) + 1)
    for i in range(len(move1), len(move2) + 1):
        window = dict(move2_items[i - len(move1) : i])
        scores[i - len(move1)] = rescale(
            similarity_score(*fill_values(move1, window)), 0.9, 0.95
        )
    return scores.tolist()


@app.post("/score")
def score_endpoint(payload: dict = Body(...)):
    move = payload.get("move")
    if not move:
        return {"error": "move is required"}

    move = {float(k): v for k, v in move.items()}

    player_scores = {}
    for player in PLAYERS:
        try:
            player_file = os.path.join("Data", f"{player}.json")
            with open(player_file, "r") as f:
                player_data = {float(k): v for k, v in json.load(f).items()}

            player_scores[player] = max(sliding_window_score(move, player_data))
        except Exception as e:
            print(f"Error processing {player}: {str(e)}")
            player_scores[player] = 0.0

    sorted_scores = dict(sorted(player_scores.items(), key=lambda x: x[1]))
    return {"scores": sorted_scores}
