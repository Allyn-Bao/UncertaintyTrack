import pprint

import pickle
import matplotlib.pyplot as plt

# Step 1: Load results.pkl
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

# results is expected to be a dict: {sequence_name: list of frame detections}
for seq, frames in results.items():
    print(f"Sequence: {seq}, Total Frames: {len(frames)}")