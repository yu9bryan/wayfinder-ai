"""推論示範：python predict.py --model crosswalk_obb_640 [--fmt pt|onnx|tflite] 圖片或資料夾 ...

結果圖存到 runs/predict/；終端機印出每張圖的偵測數與類別。
"""
import argparse

from wayfinder_obb import MODELS, load

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--fmt", default="pt", choices=["pt", "onnx", "tflite"])
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("sources", nargs="+")
a = ap.parse_args()

m = load(a.model, a.fmt)
for src in a.sources:
    for r in m.predict(src, imgsz=MODELS[a.model]["imgsz"], conf=a.conf, save=True,
                       project="runs", name="predict", exist_ok=True, verbose=False):
        boxes = r.obb if r.obb is not None else r.boxes
        names = [r.names[int(c)] for c in boxes.cls.tolist()]
        print(f"{r.path}\t{len(names)} 個\t{', '.join(names)}")
