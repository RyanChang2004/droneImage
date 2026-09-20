import torch
from exif_gps import build_image_records, build_candidate_pairs
from matching import init_matcher, match_pair

DATASET_DIR = r"D:\droneImage\droneImageAnalysis\data\dataset\Dataset1_SanPedroRiver_20230621\Dataset1_SanPedroRiver_20230621"
MAX_DIST_M = 50.0

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

    # 開始匹配
    print("\n開始特徵匹配...")
    results = []
    for i, j, dist in pairs:
        result = match_pair(
            extractor, matcher, device,
            records[i]["path"],
            records[j]["path"]
        )
        n = result["n_matches"]
        results.append((records[i]["name"], records[j]["name"], dist, n))
        print(f"  {records[i]['name']} ↔ {records[j]['name']} | GPS距離 {dist:.1f}m | 匹配點 {n}")

    # 統計
    good = [r for r in results if r[3] >= 50]
    print(f"\n完成！共處理 {len(results)} 對，其中 {len(good)} 對匹配點 ≥ 50")