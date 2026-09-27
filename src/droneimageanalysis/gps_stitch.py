import time
import cv2
import numpy as np
from pathlib import Path
from exif_gps import build_image_records
from metrics import evaluate_stitching_metrics, save_metrics_txt

DATASET_DIR = "/home/ryan0916/droneImage/data/dataset/Dataset1_SanPedroRiver_20230621/Dataset1_SanPedroRiver_20230621"
SCALE = 0.1
OUTPUT_DIR = Path("outputs")


def gps_stitch(records, scale=0.1):
    start_time = time.perf_counter()

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
    print(f"畫布大小：{canvas_w} x {canvas_h}")

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

        px = int((record["x"] - x_min) / gsd * scale)
        py = int((y_max - record["y"]) / gsd * scale)

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
    result = canvas / weight_out[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 計算 metrics（GPS 拼接沒有 Homography，pairs_data 和 loops 為空）
    metrics_df = evaluate_stitching_metrics(
        pairs_data=[],
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

    return result, metrics_df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    records = build_image_records(DATASET_DIR)
    print(f"載入 {len(records)} 張圖片")

    print("開始 GPS 拼接...")
    mosaic, metrics_df = gps_stitch(records, scale=SCALE)

    result_path = OUTPUT_DIR / "mosaic_gps.png"
    cv2.imwrite(str(result_path), mosaic)
    save_metrics_txt(metrics_df, result_path)
    metrics_df.to_csv(OUTPUT_DIR / "stitching_metrics.csv", index=False)

    print(f"\n完成！輸出到 {OUTPUT_DIR}/")
    print(metrics_df)