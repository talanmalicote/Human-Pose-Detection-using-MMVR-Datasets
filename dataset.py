import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MMVRDataset(Dataset):
    def __init__(self, root_dir, min_val=None, max_val=None):
        self.root = root_dir

        self.files = sorted([
            f.replace("_pose.npz", "")
            for f in os.listdir(root_dir)
            if "pose" in f
        ])
        # self.root = root_dir # This line is redundant

        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        base = self.files[idx]

        pose_path = os.path.join(self.root, base + "_pose.npz")
        radar_path = os.path.join(self.root, base + "_radar.npz")

        pose = np.load(os.path.join(self.root, base + "_pose.npz"))
        radar = np.load(os.path.join(self.root, base + "_radar.npz"))

        keypoints = pose[pose.files[0]]
        radar_data = radar[radar.files[0]]

        # simplify: take first person
        keypoints = keypoints[0]        # (17, 3)
        keypoints = keypoints[:, :2]    # keep only (x, y)
        keypoints = keypoints.reshape(-1)  # (34,)

        radar_data = torch.tensor(radar_data).float()
        keypoints = torch.tensor(keypoints).float()

        # Apply normalization if min_val and max_val are provided
        if self.min_val is not None and self.max_val is not None:
            # Avoid division by zero if max_val equals min_val
            if (self.max_val - self.min_val) == 0:
                keypoints = keypoints - self.min_val # Center at 0 if range is 0
            else:
                keypoints = (keypoints - self.min_val) / (self.max_val - self.min_val)

        return radar_data, keypoints
