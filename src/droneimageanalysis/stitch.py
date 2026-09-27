import json
import numpy as np
import cv2
from blending import warp_and_blend

with open("good_pairs.json") as f:
    data = json.load(f)

records = data["records"]
good_pairs = []
for p in data["pairs"]:
    good_pairs.append({
        "i": p["i"],
        "j": p["j"],
        "dist": p["dist"],
        "n_matches": p["n_matches"],
        "n_inliers": p["n_inliers"],
        "H": np.array(p["H"]),
    })

print(f"載入 {len(records)} 張圖片，{len(good_pairs)} 對有效配對")

print("開始拼接...")
mosaic = warp_and_blend(records, good_pairs)
if mosaic is not None:
    cv2.imwrite("mosaic.png", mosaic)
    print("完成！輸出到 mosaic.png")