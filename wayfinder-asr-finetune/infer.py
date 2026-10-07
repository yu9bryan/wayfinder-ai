"""單檔推論：python infer.py <音檔> [<音檔> ...]

自動從 Hugging Face 下載 Taiwan Tongues ASR CE 基底，載入 adapter/ 的 LoRA adapter。
"""
import sys
from pathlib import Path
import torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE = "adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0"
ADAPTER = Path(__file__).parent / "adapter"

if len(sys.argv) < 2:
    sys.exit(__doc__)

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" else torch.float32
proc = WhisperProcessor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=dtype)
model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload().to(dev).eval()

for f in sys.argv[1:]:
    wav, _ = librosa.load(f, sr=16000)
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats.to(dev, dtype), language="zh", task="transcribe", max_new_tokens=128)
    print(f"{f}\t{proc.batch_decode(ids, skip_special_tokens=True)[0]}")
