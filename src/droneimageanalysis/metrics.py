import time
import numpy as np
import pandas as pd
from pathlib import Path


# ── 1. Reprojection Error ─────────────────────────────────────
def compute_reprojection_error(pairs_data: list[dict]) -> float:
    """
    pairs_data: list of {H, kpts_a, kpts_b, matches}
    只用 RANSAC inliers 計算 global RMSE
    """
    all_errors = []
    for pair in pairs_data:
        H = pair["H"].astype(np.float64)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        kpts_a = pair["kpts_a"]
        kpts_b = pair["kpts_b"]
        inlier_mask = pair["inlier_mask"]
        matches = pair["matches"]

        for k, (mi, mj) in enumerate(matches):
            if not inlier_mask[k]:
                continue
            p = np.array([kpts_a[mi][0], kpts_a[mi][1], 1.0], dtype=np.float64)
            q = np.array([kpts_b[mj][0], kpts_b[mj][1]], dtype=np.float64)
            p_proj = H @ p
            w = p_proj[2]
            if abs(w) < 1e-12:
                continue
            p_proj = p_proj[:2] / w
            err = np.linalg.norm(p_proj - q)
            all_errors.append(err ** 2)

    if not all_errors:
        return np.nan
    return float(np.sqrt(np.mean(all_errors)))


# ── 2. Inlier Ratio ───────────────────────────────────────────
def compute_inlier_ratio(pairs_data: list[dict]) -> float:
    total_matches = sum(len(p["matches"]) for p in pairs_data)
    total_inliers = sum(int(p["inlier_mask"].sum()) for p in pairs_data)
    if total_matches == 0:
        return np.nan
    return total_inliers / total_matches


# ── 3. Inlier Count ───────────────────────────────────────────
def compute_inlier_count(pairs_data: list[dict]) -> int:
    return sum(int(p["inlier_mask"].sum()) for p in pairs_data)


# ── 4. Cycle / Loop Error ─────────────────────────────────────
def compute_cycle_loop_error(loops: list[list[np.ndarray]], img_w: int, img_h: int) -> float:
    """
    loops: list of [H_A_to_B, H_B_to_C, ..., H_N_to_A]
    """
    if not loops:
        return np.nan

    ref_pts = np.array([
        [0, 0],
        [img_w - 1, 0],
        [img_w - 1, img_h - 1],
        [0, img_h - 1],
        [img_w / 2, img_h / 2],
    ], dtype=np.float64)

    loop_rmses = []
    for chain in loops:
        H_cycle = np.eye(3, dtype=np.float64)
        for H in chain:
            H = H.astype(np.float64)
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]
            H_cycle = H @ H_cycle

        errors = []
        for p in ref_pts:
            ph = np.array([p[0], p[1], 1.0])
            p_proj = H_cycle @ ph
            w = p_proj[2]
            if abs(w) < 1e-12:
                continue
            p_proj = p_proj[:2] / w
            err = np.linalg.norm(p_proj - p)
            errors.append(err ** 2)

        if errors:
            loop_rmses.append(float(np.sqrt(np.mean(errors))))

    if not loop_rmses:
        return np.nan
    return float(np.mean(loop_rmses))


# ── 5. Seam Error ─────────────────────────────────────────────
def compute_seam_error(canvas: np.ndarray, weight: np.ndarray) -> float:
    """
    在重疊區域（weight > 1）計算像素差異
    """
    try:
        overlap_mask = weight > 1.0
        if not np.any(overlap_mask):
            return np.nan

        # 正規化後的圖
        normalized = canvas / np.maximum(weight[:, :, None], 1e-6)
        normalized = np.clip(normalized, 0, 255)

        # 重疊區域的變異數作為接縫誤差
        overlap_pixels = normalized[overlap_mask]
        seam_err = float(np.std(overlap_pixels) / 255.0)
        return seam_err
    except Exception:
        return np.nan


# ── 6. Distortion ─────────────────────────────────────────────
def compute_distortion(pairs_data: list[dict]) -> float:
    """
    衡量 Homography 偏離純平移的程度
    H ≈ [[1,0,tx],[0,1,ty],[0,0,1]] 時 distortion ≈ 0
    """
    if not pairs_data:
        return np.nan

    distortions = []
    for pair in pairs_data:
        H = pair["H"].astype(np.float64)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        # 移除平移部分，看剩下的偏差
        H_no_t = H.copy()
        H_no_t[0, 2] = 0
        H_no_t[1, 2] = 0
        diff = np.linalg.norm(H_no_t - np.eye(3))
        distortions.append(diff)

    if not distortions:
        return np.nan
    return float(np.mean(distortions))


# ── 主函式 ────────────────────────────────────────────────────
def evaluate_stitching_metrics(
    pairs_data: list[dict],
    records: list[dict],
    successful_indices: set,
    failed_indices: list[int],
    canvas: np.ndarray,
    weight: np.ndarray,
    loops: list[list[np.ndarray]],
    img_w: int,
    img_h: int,
    start_time: float,
) -> pd.DataFrame:

    input_image_count = len(records)
    successful_image_count = len(successful_indices)
    failed_image_count = input_image_count - successful_image_count
    stitch_success_rate = (
        successful_image_count / input_image_count
        if input_image_count > 0 else np.nan
    )

    # pipeline_status
    if successful_image_count == input_image_count:
        pipeline_status = "success"
    elif successful_image_count >= 2:
        pipeline_status = "partial_success"
    else:
        pipeline_status = "failed"

    total_processing_time_sec = time.perf_counter() - start_time
    avg_processing_time_per_image_sec = (
        total_processing_time_sec / input_image_count
        if input_image_count > 0 else np.nan
    )

    reprojection_error_px = compute_reprojection_error(pairs_data)
    inlier_ratio = compute_inlier_ratio(pairs_data)
    inlier_count = compute_inlier_count(pairs_data)
    cycle_loop_error_px = compute_cycle_loop_error(loops, img_w, img_h)
    seam_error = compute_seam_error(canvas, weight)
    distortion = compute_distortion(pairs_data)

    # logging
    def fmt(v):
        return "N/A" if (isinstance(v, float) and np.isnan(v)) else v

    print("\n=== Stitching Summary ===")
    print(f"Pipeline Status    : {pipeline_status}")
    print(f"Input Images       : {input_image_count}")
    print(f"Successful Images  : {successful_image_count}")
    print(f"Failed Images      : {failed_image_count}")
    print(f"Success Rate       : {stitch_success_rate:.2%}" if not np.isnan(stitch_success_rate) else "Success Rate       : N/A")
    print(f"Failed Indices     : {failed_indices}")
    print(f"Total Time         : {total_processing_time_sec:.2f} sec")
    print(f"Avg Time / Image   : {avg_processing_time_per_image_sec:.4f} sec" if not np.isnan(avg_processing_time_per_image_sec) else "Avg Time / Image   : N/A")
    print("\n=== Stitching Metrics ===")
    print(f"Reprojection Error : {fmt(reprojection_error_px)} px")
    print(f"Inlier Ratio       : {fmt(inlier_ratio)}")
    print(f"Inlier Count       : {inlier_count}")
    print(f"Cycle / Loop Error : {fmt(cycle_loop_error_px)} px")
    print(f"Seam Error         : {fmt(seam_error)}")
    print(f"Distortion         : {fmt(distortion)}")
    print("=========================")

    metrics_df = pd.DataFrame([{
        "pipeline_status": pipeline_status,
        "input_image_count": input_image_count,
        "successful_image_count": successful_image_count,
        "failed_image_count": failed_image_count,
        "stitch_success_rate": stitch_success_rate,
        "failed_image_indices": failed_indices,
        "total_processing_time_sec": round(total_processing_time_sec, 4),
        "avg_processing_time_per_image_sec": round(avg_processing_time_per_image_sec, 4) if not np.isnan(avg_processing_time_per_image_sec) else np.nan,
        "reprojection_error_px": reprojection_error_px,
        "inlier_ratio": inlier_ratio,
        "inlier_count": inlier_count,
        "cycle_loop_error_px": cycle_loop_error_px,
        "seam_error": seam_error,
        "distortion": distortion,
    }])

    return metrics_df


# ── 儲存 metrics.txt ──────────────────────────────────────────
def save_metrics_txt(metrics_df: pd.DataFrame, result_image_path) -> Path:
    result_image_path = Path(result_image_path)
    metrics_path = result_image_path.parent / "metrics.txt"
    try:
        metrics_path.write_text(
            metrics_df.to_string(index=False),
            encoding="utf-8",
        )
    except IOError as e:
        raise IOError(f"寫入 metrics.txt 失敗：{e}")
    return metrics_path