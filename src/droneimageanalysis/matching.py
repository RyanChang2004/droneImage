import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def init_matcher(device: torch.device):
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)  # 從2048降到1024
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def match_pair(extractor, matcher, device, path_a: str, path_b: str) -> dict:
    # resize=1024 限制長邊，縮小圖片加速
    image_a = load_image(path_a, resize=1024).to(device)
    image_b = load_image(path_b, resize=1024).to(device)

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