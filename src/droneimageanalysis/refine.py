import time
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from exif_gps import build_image_records
from matching import init_matcher, match_pair
from metrics import evaluate_stitching_metrics, save_metrics_txt
import torch

DATASET_DIR = "/home/ryan0916/droneImage/data/dataset/Dataset1_SanPedroRiver_20230621/Dataset1_SanPedroRiver_20230621"
SCALE = 0.1
OUTPUT_DIR = Path("outputs")


def estimate_translation(kpts_a, kpts_b, matches, min_inliers=4):
    if len(matches) < min_inliers:
        return None, None, None
    pts_a = kpts_a[matches[:, 0]].astype(np.float64)
    pts_b = kpts_b[matches[:, 1]].astype(np.float64)
    dx = np.median(pts_b[:, 0] - pts_a[:, 0])
    dy = np.median(pts_b[:, 1] - pts_a[:, 1])

    # 建立 inlier_mask（距離中位數 ±10px 內的算 inlier）
    diffs = pts_b - pts_a
    dist = np.sqrt((diffs[:, 0] - dx)**2 + (diffs[:, 1] - dy)**2)
    inlier_mask = dist < 10.0

    # 建立簡單的平移 Homography
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy

    return dx, dy, H, inlier_mask


def gps_to_pixel(record, x_min, y_max, gsd, scale):
    px = int((record["x"] - x_min) / gsd * scale)
    py = int((y_max - record["y"]) / gsd * scale)
    return px, py


def refine_and_stitch(records, scale=0.1):
    start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    extractor, matcher = init_matcher(device)

    sample_img = cv2.imread(records[0]["path"])
    img_w_orig = sample_img.shape[1]
    img_h_orig = sample_img.shape[0]
    alt = records[0]["alt"]
    fov = 84
    gsd = (2 * alt * np.tan(np.radians(fov / 2))) / img_w_orig

    x_coords = [r["x"] for r in records]
    y_coords = [r["y"] for r in records]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    img_h_m = 2 * alt * np.tan(np.radians(fov / 2)) * img_h_orig / img_w_orig
    canvas_w = int((x_max - x_min + img_w_orig * gsd) / gsd * scale)
    canvas_h = int((y_max - y_min + img_h_m) / gsd * scale)

    positions = {}
    for idx, record in enumerate(records):
        px, py = gps_to_pixel(record, x_min, y_max, gsd, scale)
        positions[idx] = [px, py]

    # 視覺精修
    print("開始視覺精修...")
    translations = defaultdict(list)
    pairs_data = []

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dx_m = abs(records[i]["x"] - records[j]["x"])
            dy_m = abs(records[i]["y"] - records[j]["y"])
            dist = (dx_m**2 + dy_m**2) ** 0.5
            if dist > 50.0:
                continue

            result = match_pair(
                extractor, matcher, device,
                records[i]["path"],
                records[j]["path"]
            )

            if result["n_matches"] < 10:
                continue

            out = estimate_translation(
                result["kpts_a"],
                result["kpts_b"],
                result["matches"]
            )

            if out[0] is None:
                continue

            dx, dy, H, inlier_mask = out

            # 收集 metrics 用的資料
            pairs_data.append({
                "H": H,
                "kpts_a": result["kpts_a"],
                "kpts_b": result["kpts_b"],
                "matches": result["matches"],
                "inlier_mask": inlier_mask,
            })

            translations[j].append((
                positions[i][0] + int(dx * scale),
                positions[i][1] + int(dy * scale)
            ))
            print(f"  {records[i]['name']} ↔ {records[j]['name']} | 平移 ({dx:.1f}, {dy:.1f})")

    # 用中位數更新位置
    for idx in range(len(records)):
        if idx in translations and len(translations[idx]) > 0:
            xs = [t[0] for t in translations[idx]]
            ys = [t[1] for t in translations[idx]]
            positions[idx][0] = int(np.median(xs))
            positions[idx][1] = int(np.median(ys))

    # 拼接
    print("開始拼接...")
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    successful_indices = set()
    failed_indices = []

    for idx, record in enumerate(records):
        img = cv2.imread(record["path"])
        if img is None:
            failed_indices.append(idx)
            continue

        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = img.astype(np.float32)
        h, w = img.shape[:2]

        px, py = positions[idx]
        x1 = px - w // 2
        y1 = py - h // 2
        x2 = x1 + w
        y2 = y1 + h

        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(canvas_w, x2)
        cy2 = min(canvas_h, y2)

        ix1 = cx1 - x1
        iy1 = cy1 - y1
        ix2 = ix1 + (cx2 - cx1)
        iy2 = iy1 + (cy2 - cy1)

        if cx2 <= cx1 or cy2 <= cy1:
            failed_indices.append(idx)
            continue

        mask = cv2.distanceTransform(
            np.ones((h, w), dtype=np.uint8), cv2.DIST_L2, 5
        )
        mask = mask / mask.max()

        canvas[cy1:cy2, cx1:cx2] += img[iy1:iy2, ix1:ix2] * mask[iy1:iy2, ix1:ix2, None]
        weight[cy1:cy2, cx1:cx2] += mask[iy1:iy2, ix1:ix2]
        successful_indices.add(idx)

    weight_out = np.maximum(weight, 1e-6)
    result_img = canvas / weight_out[:, :, None]
    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    metrics_df = evaluate_stitching_metrics(
        pairs_data=pairs_data,
        records=records,
        successful_indices=successful_indices,
        failed_indices=failed_indices,
        canvas=canvas,
        weight=weight,
        loops=[],
        img_w=img_w_orig,
        img_h=img_h_orig,
        start_time=start_time,
    )

    return result_img, metrics_df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = build_image_records(DATASET_DIR)
    records = records[:50]
    print(f"載入 {len(records)} 張圖片")

    mosaic, metrics_df = refine_and_stitch(records, scale=SCALE)

    result_path = OUTPUT_DIR / "mosaic_refined.png"
    cv2.imwrite(str(result_path), mosaic)
    save_metrics_txt(metrics_df, result_path)
    metrics_df.to_csv(OUTPUT_DIR / "stitching_metrics_refined.csv", index=False)

    print(f"\n完成！輸出到 {OUTPUT_DIR}/")
    print(metrics_df)