import cv2
import numpy as np


def compute_homography(kpts_a, kpts_b, matches, min_matches: int = 10):
    if len(matches) < min_matches:
        return None, None

    # 取出有匹配的特徵點座標
    pts_a = kpts_a[matches[:, 0]]
    pts_b = kpts_b[matches[:, 1]]

    # 用 RANSAC 計算 Homography
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        return None, None

    n_inliers = int(mask.sum())
    return H, n_inliers