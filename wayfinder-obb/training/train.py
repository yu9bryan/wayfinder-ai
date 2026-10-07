"""訓練：python training/train.py --model crosswalk_obb_640 [--cfg configs/train.yaml] [--data 你的資料集.yaml]

--model 選 configs/train.yaml 裡的一組設定（crosswalk_obb_640、crosswalk_obb_320、traffic_light_640）。
會先套用與釋出權重相同的 head patch（窄化 box 分支），再從架構 YAML 從頭訓練。
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wayfinder_obb import apply_patch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cfg", default=str(ROOT / "configs/train.yaml"))
ap.add_argument("--data", default=None, help="覆寫 train.yaml 裡的 data")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="runs/train")
a = ap.parse_args()

c = yaml.safe_load(open(a.cfg, encoding="utf-8"))[a.model]
apply_patch(a.model)
from ultralytics import YOLO  # noqa: E402

m = YOLO(str(ROOT / c["cfg"]), task=c["task"])
n = sum(p.numel() for p in m.model.parameters())
h = sum(p.numel() for p in m.model.model[-1].parameters())
print(f"[{a.model}] params={n:,}  head={h:,} ({100 * h / n:.1f}%)")

m.train(
    data=a.data or str(ROOT / c["data"]),
    epochs=a.epochs or c["epochs"],
    batch=a.batch or c["batch"],
    imgsz=c["imgsz"],
    lr0=c["lr0"],
    cos_lr=c["cos_lr"],
    seed=c["seed"],
    workers=a.workers,
    device=a.device,
    amp=True,
    patience=0,
    project=a.project,
    name=a.model,
    exist_ok=True,
)
