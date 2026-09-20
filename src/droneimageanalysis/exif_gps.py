from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pyproj import Transformer


def get_gps_from_exif(image_path: str) -> dict | None:
    """從圖片 EXIF 讀取 GPS 資訊"""
    img = Image.open(image_path)
    exif_data = img._getexif()
    if not exif_data:
        return None

    gps_info = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_info[gps_tag] = gps_value

    if not gps_info:
        return None

    def dms_to_decimal(dms, ref):
        d, m, s = dms
        decimal = float(d) + float(m) / 60 + float(s) / 3600
        if ref in ["S", "W"]:
            decimal = -decimal
        return decimal

    lat = dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
    lon = dms_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
    alt = float(gps_info.get("GPSAltitude", 0))

    return {"lat": lat, "lon": lon, "alt": alt}


def gps_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 經緯度轉 UTM 平面座標（公尺）"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def build_image_records(dataset_dir: str) -> list[dict]:
    """掃描資料夾，建立每張圖片的資訊清單"""
    import os
    records = []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue
        path = os.path.join(dataset_dir, fname)
        gps = get_gps_from_exif(path)
        if gps is None:
            print(f"[跳過] {fname} 沒有 GPS 資訊")
            continue
        x, y = gps_to_utm(gps["lat"], gps["lon"])
        records.append({
            "path": path,
            "name": fname,
            "lat": gps["lat"],
            "lon": gps["lon"],
            "alt": gps["alt"],
            "x": x,
            "y": y,
        })
        print(f"[載入] {fname} → UTM ({x:.1f}, {y:.1f}), 高度 {gps['alt']:.1f}m")
    return records


def build_candidate_pairs(records: list[dict], max_dist_m: float = 50.0) -> list[tuple]:
    """用 GPS 距離篩選出可能有重疊的圖片對"""
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            dx = records[i]["x"] - records[j]["x"]
            dy = records[i]["y"] - records[j]["y"]
            dist = (dx**2 + dy**2) ** 0.5
            if dist <= max_dist_m:
                pairs.append((i, j, dist))
    pairs.sort(key=lambda x: x[2])
    print(f"\n找到 {len(pairs)} 對候選圖片（距離 ≤ {max_dist_m}m）")
    return pairs