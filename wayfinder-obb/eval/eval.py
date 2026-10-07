"""評估：python eval/eval.py --model crosswalk_obb_640 --data 你的資料集.yaml [--fmt pt|onnx|tflite] [--split test]

輸出 mAP50、mAP50-95、Precision、Recall；紅綠燈另外輸出各類別 AP50。
另可加 --by-size 依目標大小（短邊 < 24 px、24–64 px、≥ 64 px）分段計算召回率（只支援一般偵測框）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wayfinder_obb import MODELS, load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--data", required=True)
ap.add_argument("--fmt", default="pt", choices=["pt", "onnx", "tflite"])
ap.add_argument("--split", default="test")
ap.add_argument("--device", default="cpu")
ap.add_argument("--by-size", action="store_true")
a = ap.parse_args()

m = load(a.model, a.fmt)
imgsz = MODELS[a.model]["imgsz"]
r = m.val(data=a.data, split=a.split, imgsz=imgsz, batch=1 if a.fmt != "pt" else 16,
          device=a.device, plots=False, verbose=False)
b = r.box
out = {"model": a.model, "fmt": a.fmt, "split": a.split, "imgsz": imgsz,
       "mAP50": round(float(b.map50), 5), "mAP50_95": round(float(b.map), 5),
       "precision": round(float(b.mp), 5), "recall": round(float(b.mr), 5)}
if len(r.names) > 1:
    out["AP50_per_class"] = {r.names[i]: round(float(v), 5) for i, v in enumerate(b.ap50)}

if a.by_size:
    if MODELS[a.model]["task"] != "detect":
        sys.exit("--by-size 只支援一般偵測框（traffic_light_640）")
    import glob, os
    import numpy as np
    import yaml
    from ultralytics.utils.metrics import box_iou
    import torch

    d = yaml.safe_load(open(a.data, encoding="utf-8"))
    img_dir = Path(d.get("path", "")) / d[a.split]
    bands = {"small(<24px)": [0, 0], "medium(24-64px)": [0, 0], "large(>=64px)": [0, 0]}
    for img in sorted(glob.glob(str(img_dir / "*"))):
        lbl = img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl) or os.path.getsize(lbl) == 0:
            continue
        p = m.predict(img, imgsz=imgsz, conf=0.25, device=a.device, verbose=False)[0]
        h, w = p.orig_shape
        gt = np.loadtxt(lbl, ndmin=2)
        s = imgsz / max(h, w)
        xyxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                         (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h], 1)
        iou = box_iou(torch.tensor(xyxy, dtype=torch.float32), p.boxes.xyxy.cpu()) if len(p.boxes) else None
        for i, g in enumerate(gt):
            size = (g[3] * w * g[4] * h) ** 0.5 * s
            k = "small(<24px)" if size < 24 else "medium(24-64px)" if size < 64 else "large(>=64px)"
            bands[k][0] += 1
            if iou is not None and float(iou[i].max()) >= 0.5:
                bands[k][1] += 1
    out["recall_by_size"] = {k: round(hit / n, 5) if n else None for k, (n, hit) in bands.items()}

print(json.dumps(out, ensure_ascii=False, indent=1))
