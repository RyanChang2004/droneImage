import cv2
import numpy as np


def compute_homography(kpts_a, kpts_b, matches, min_matches: int = 10):
    """
    用匹配點計算 Homography 矩陣
    kpts_a, kpts_b: 特徵點座標
    matches: 匹配對應關係
    回傳 H（變換矩陣）和 mask（哪些點是 inlier）
    """
    if len(matches) < min_matches:
        return None, None

    # 取出有匹配的特徵點座標
    pts_a = kpts_a[matches[:, 0]].numpy()
    pts_b = kpts_b[matches[:, 1]].numpy()

    # 用 RANSAC 計算 Homography，過濾掉錯誤的匹配點
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None:
        return None, None

    n_inliers = mask.sum()
    return H, int(n_inliers)