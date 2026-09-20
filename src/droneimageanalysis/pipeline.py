import torch
from exif_gps import build_image_records, build_candidate_pairs
from matching import init_matcher, match_pair
from homography import compute_homography
from blending import warp_and_blend
import cv2

DATASET_DIR="/home/ryan0916/droneImage/data/dataset/Dataset1_SanPedroRiver_20230621/Dataset1_SanPedroRiver_20230621"
MAX_DIST_M = 30.0
MIN_MATCHES = 30

if __name__ == "__main__":
    # 建立圖片清單
    records = build_image_records(DATASET_DIR)
    if not records:
        print("沒有找到任何有 GPS 的圖片")
        exit()
    print(f"\n共載入 {len(records)} 張圖片")

    # 找候選對
    pairs = build_candidate_pairs(records, max_dist_m=MAX_DIST_M)
    if not pairs:
        print("沒有找到任何候選圖片對，試著調大 MAX_DIST_M")
        exit()

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用裝置：{device}")
    extractor, matcher = init_matcher(device)

    # 開始匹配 + Homography
    print("\n開始特徵匹配 + Homography 計算...")
    good_pairs = []

    for i, j, dist in pairs:
        result = match_pair(
            extractor, matcher, device,
            records[i]["path"],
            records[j]["path"]
        )
        n = result["n_matches"]

        if n < MIN_MATCHES:
            continue

        H, n_inliers = compute_homography(
            result["kpts_a"],
            result["kpts_b"],
            result["matches"]
        )

        if H is None:
            continue

        good_pairs.append({
            "i": i,
            "j": j,
            "dist": dist,
            "n_matches": n,
            "n_inliers": n_inliers,
            "H": H,
        })

        print(f"  {records[i]['name']} ↔ {records[j]['name']} | "
              f"GPS距離 {dist:.1f}m | 匹配點 {n} | inliers {n_inliers}")

    print(f"\n完成！共找到 {len(good_pairs)} 對有效配對")

    print("\n開始拼接...")
    mosaic = warp_and_blend(records, good_pairs)
    output_path = "mosaic.png"
    cv2.imwrite(output_path, mosaic)
    print(f"完成！輸出到 {output_path}")