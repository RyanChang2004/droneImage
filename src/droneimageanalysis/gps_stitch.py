import cv2
import numpy as np
from exif_gps import build_image_records

DATASET_DIR = "/home/ryan0916/droneImage/data/dataset/Dataset1_SanPedroRiver_20230621/Dataset1_SanPedroRiver_20230621"
SCALE = 0.1  # 圖片縮放比例


def gps_stitch(records, scale=0.1):
    # 從高度估算每個像素代表多少公尺
    # DJI 相機水平視角約 84 度
    sample_img = cv2.imread(records[0]["path"])
    img_w = sample_img.shape[1]
    alt = records[0]["alt"]
    fov = 84  # 度
    gsd = (2 * alt * np.tan(np.radians(fov / 2))) / img_w  # 公尺/像素
    print(f"飛行高度：{alt:.1f}m，GSD：{gsd:.3f} 公尺/像素")

    # 計算每張圖在畫布上的位置
    x_coords = [r["x"] for r in records]
    y_coords = [r["y"] for r in records]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 圖片實際大小（公尺）
    img_h_m = 2 * alt * np.tan(np.radians(fov / 2)) * sample_img.shape[0] / img_w

    # 畫布大小（像素）
    canvas_w = int((x_max - x_min + img_w * gsd) / gsd * scale)
    canvas_h = int((y_max - y_min + img_h_m) / gsd * scale)
    print(f"畫布大小：{canvas_w} x {canvas_h}")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for record in records:
        img = cv2.imread(record["path"])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = img.astype(np.float32)
        h, w = img.shape[:2]

        # GPS 座標轉畫布像素位置
        px = int((record["x"] - x_min) / gsd * scale)
        py = int((y_max - record["y"]) / gsd * scale)  # y 軸翻轉

        # 貼圖位置（圖片中心對齊 GPS 座標）
        x1 = px - w // 2
        y1 = py - h // 2
        x2 = x1 + w
        y2 = y1 + h

        # 確保不超出畫布
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(canvas_w, x2)
        cy2 = min(canvas_h, y2)

        ix1 = cx1 - x1
        iy1 = cy1 - y1
        ix2 = ix1 + (cx2 - cx1)
        iy2 = iy1 + (cy2 - cy1)

        if cx2 <= cx1 or cy2 <= cy1:
            continue

        # feather blending
        mask = cv2.distanceTransform(
            np.ones((h, w), dtype=np.uint8), cv2.DIST_L2, 5
        )
        mask = mask / mask.max()

        canvas[cy1:cy2, cx1:cx2] += img[iy1:iy2, ix1:ix2] * mask[iy1:iy2, ix1:ix2, None]
        weight[cy1:cy2, cx1:cx2] += mask[iy1:iy2, ix1:ix2]

    weight = np.maximum(weight, 1e-6)
    result = canvas / weight[:, :, None]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


if __name__ == "__main__":
    records = build_image_records(DATASET_DIR)
    records = records[:50]  # 先用 50 張
    print(f"載入 {len(records)} 張圖片")

    print("開始 GPS 拼接...")
    mosaic = gps_stitch(records, scale=SCALE)
    cv2.imwrite("mosaic_gps.png", mosaic)
    print("完成！輸出到 mosaic_gps.png")