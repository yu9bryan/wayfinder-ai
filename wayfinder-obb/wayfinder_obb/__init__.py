"""Wayfinder 斑馬線 OBB／紅綠燈偵測模型的載入器。

    from wayfinder_obb import load
    model = load("crosswalk_obb_640")        # 回傳 ultralytics.YOLO
    results = model("street.jpg", imgsz=640)

head patch 會改 Ultralytics 類別本身，同一個 process 只能載入同一種 patch 設定的 .pt。
要同時跑斑馬線和紅綠燈，請改用 ONNX 或 TFLite（計算圖已固化，不需要 patch）。
"""
from pathlib import Path

from .heads import patch_detect_head, patch_obb_head

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "crosswalk_obb_640": dict(task="obb", imgsz=640, head=("obb", dict(c2=16, dw=True))),
    "crosswalk_obb_320": dict(task="obb", imgsz=320, head=("obb", dict(c2=16))),
    "traffic_light_640": dict(task="detect", imgsz=640, head=("detect", dict(c2=16, dw=True))),
}
_applied = None


def apply_patch(name: str):
    """依模型名稱套用對應的 head patch。訓練與匯出腳本也呼叫這個函式。"""
    global _applied
    head = MODELS[name]["head"]
    if _applied is not None and _applied != head:
        raise RuntimeError("同一個 process 已套用另一種 head patch；請另開 process，或改用 onnx／tflite。")
    kind, kw = head
    (patch_obb_head if kind == "obb" else patch_detect_head)(**kw)
    _applied = head


def load(name: str, fmt: str = "pt"):
    """載入模型。fmt：pt（自動 patch）、onnx、tflite（fp16）。"""
    if name not in MODELS:
        raise KeyError(f"未知模型 {name}，可用：{', '.join(MODELS)}")
    from ultralytics import YOLO

    suffix = {"pt": ".pt", "onnx": ".onnx", "tflite": "_fp16.tflite"}[fmt]
    if fmt == "pt":
        apply_patch(name)
    return YOLO(str(ROOT / "models" / f"{name}{suffix}"), task=MODELS[name]["task"])


__all__ = ["load", "apply_patch", "MODELS", "patch_obb_head", "patch_detect_head"]
