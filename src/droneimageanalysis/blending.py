import cv2
import numpy as np


def warp_and_blend(records, good_pairs):
    """
    把所有圖片拼接成一張大圖
    """
    n = len(records)
    
    # 用第一張圖當基準，計算每張圖的全域變換矩陣
    global_H = [None] * n
    global_H[0] = np.eye(3)  # 第一張圖不需要變換

    # 從 good_pairs 建立圖（哪些圖片有連接）
    for pair in good_pairs:
        i, j = pair["i"], pair["j"]
        H = pair["H"]
        if global_H[i] is not None and global_H[j] is None:
            global_H[j] = global_H[i] @ np.linalg.inv(H)
        elif global_H[j] is not None and global_H[i] is None:
            global_H[i] = global_H[j] @ H

    # 計算畫布大小
    corners_list = []
    for idx, record in enumerate(records):
        if global_H[idx] is None:
            continue
        img = cv2.imread(record["path"])
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners = cv2.perspectiveTransform(corners[None], global_H[idx])[0]
        corners_list.append(corners)

    all_corners = np.concatenate(corners_list)
    x_min, y_min = all_corners.min(axis=0).astype(int)
    x_max, y_max = all_corners.max(axis=0).astype(int)

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"畫布大小：{canvas_w} x {canvas_h}")

    # 偏移矩陣（讓所有座標變成正數）
    offset = np.array([[1, 0, -x_min],
                       [0, 1, -y_min],
                       [0, 0, 1]], dtype=np.float64)

    # 建立畫布
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # 把每張圖貼上去
    for idx, record in enumerate(records):
        if global_H[idx] is None:
            continue
        img = cv2.imread(record["path"])
        img = img.astype(np.float32)
        h, w = img.shape[:2]

        H = offset @ global_H[idx]
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h))

        # feather blending 權重
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.distanceTransform(
            np.ones((h, w), dtype=np.uint8), cv2.DIST_L2, 5
        )
        mask = mask / mask.max()
        warped_mask = cv2.warpPerspective(mask, H, (canvas_w, canvas_h))

        canvas += warped * warped_mask[:, :, None]
        weight += warped_mask

    # 正規化
    weight = np.maximum(weight, 1e-6)
    result = canvas / weight[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result