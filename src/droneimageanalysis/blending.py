import cv2
import numpy as np
from collections import defaultdict, deque


def build_global_transforms(records, good_pairs):
    """用 BFS 從第一張圖出發，計算每張圖的全域變換矩陣"""
    n = len(records)

    # 建立鄰接表
    graph = defaultdict(list)
    for pair in good_pairs:
        i, j = pair["i"], pair["j"]
        H = pair["H"]
        graph[i].append((j, H))
        graph[j].append((i, np.linalg.inv(H)))

    # BFS
    global_H = [None] * n
    global_H[0] = np.eye(3)
    queue = deque([0])
    visited = set([0])

    while queue:
        curr = queue.popleft()
        for neighbor, H in graph[curr]:
            if neighbor not in visited:
                global_H[neighbor] = global_H[curr] @ H
                visited.add(neighbor)
                queue.append(neighbor)

    connected = sum(1 for h in global_H if h is not None)
    print(f"成功連接 {connected} / {n} 張圖片")
    return global_H


def warp_and_blend(records, good_pairs, scale=0.1):
    global_H = build_global_transforms(records, good_pairs)

    # 計算畫布大小
    corners_list = []
    for idx, record in enumerate(records):
        if global_H[idx] is None:
            continue
        img = cv2.imread(record["path"])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h, w = img.shape[:2]

        # 縮放矩陣
        S = np.diag([scale, scale, 1.0])
        H_scaled = S @ global_H[idx] @ np.linalg.inv(S)

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners = cv2.perspectiveTransform(corners[None], H_scaled)[0]
        corners_list.append(corners)

    if not corners_list:
        print("沒有圖片可以拼接")
        return None

    all_corners = np.concatenate(corners_list)
    x_min, y_min = all_corners.min(axis=0).astype(int)
    x_max, y_max = all_corners.max(axis=0).astype(int)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"畫布大小：{canvas_w} x {canvas_h}")

    offset = np.array([[1, 0, -x_min],
                       [0, 1, -y_min],
                       [0, 0, 1]], dtype=np.float64)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for idx, record in enumerate(records):
        if global_H[idx] is None:
            continue
        img = cv2.imread(record["path"])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = img.astype(np.float32)
        h, w = img.shape[:2]

        S = np.diag([scale, scale, 1.0])
        H_scaled = S @ global_H[idx] @ np.linalg.inv(S)
        H_final = offset @ H_scaled

        warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))

        mask = cv2.distanceTransform(
            np.ones((h, w), dtype=np.uint8), cv2.DIST_L2, 5
        )
        mask = mask / mask.max()
        warped_mask = cv2.warpPerspective(mask, H_final, (canvas_w, canvas_h))

        canvas += warped * warped_mask[:, :, None]
        weight += warped_mask

    weight = np.maximum(weight, 1e-6)
    result = canvas / weight[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result