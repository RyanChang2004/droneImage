import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def init_matcher(device: torch.device):
    """初始化 LightGlue 模型"""
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def match_pair(extractor, matcher, device, path_a: str, path_b: str) -> dict:
    """對兩張圖片做特徵匹配，回傳匹配結果"""
    image_a = load_image(path_a).to(device)
    image_b = load_image(path_b).to(device)

    feats_a = extractor.extract(image_a)
    feats_b = extractor.extract(image_b)

    result = matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, result = rbd(feats_a), rbd(feats_b), rbd(result)

    return {
        "kpts_a": feats_a["keypoints"].cpu().numpy(),
        "kpts_b": feats_b["keypoints"].cpu().numpy(),
        "matches": result["matches"].cpu().numpy(),
        "n_matches": len(result["matches"]),
    }