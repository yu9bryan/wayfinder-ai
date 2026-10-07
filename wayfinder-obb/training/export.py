"""匯出：python training/export.py --model crosswalk_obb_640 --weights runs/train/crosswalk_obb_640/weights/best.pt

pt → ONNX → TFLite fp16。輸出放在權重同一個資料夾。

本 repo 預設不匯出 int8：兩種模型的偵測頭都含 DFL 分佈回歸，全整數量化後精度大幅下降
（實測斑馬線 mAP50 0.818 → 0.098、紅綠燈 0.982 → 0.874）。如仍要試，加 --int8 並以 eval/eval.py 重新驗證。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wayfinder_obb import MODELS, apply_patch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--weights", default=None, help="預設為 models/<model>.pt")
ap.add_argument("--int8", action="store_true", help="另外匯出 int8 TFLite（需 --data 做校準）")
ap.add_argument("--data", default=None)
a = ap.parse_args()

cfg = MODELS[a.model]
w = a.weights or str(ROOT / "models" / f"{a.model}.pt")
apply_patch(a.model)
from ultralytics import YOLO  # noqa: E402

onnx = YOLO(w, task=cfg["task"]).export(format="onnx", imgsz=cfg["imgsz"], device="cpu", simplify=True)
fp16 = YOLO(w, task=cfg["task"]).export(format="tflite", imgsz=cfg["imgsz"], device="cpu", half=True)
print("ONNX        ", onnx)
print("TFLite fp16 ", fp16)
if a.int8:
    q = YOLO(w, task=cfg["task"]).export(format="tflite", imgsz=cfg["imgsz"], device="cpu", int8=True, data=a.data)
    print("TFLite int8 ", q, "（請務必重新評估）")
