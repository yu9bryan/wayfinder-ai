"""CER 評估（微調前後共用）。--adapter 給 LoRA 輸出目錄就評估微調後模型。

用法：
  python eval_asr.py --corpus <dataset 目錄> --tag baseline
  python eval_asr.py --corpus <dataset 目錄> --adapter out/lora-xxx --tag lora
"""
import os as _os
ROOT = _os.environ.get("TTASR_ROOT", ".")  # 工作區根目錄（預設為倉庫根目錄），見 README
import argparse, re, time, unicodedata, json
from pathlib import Path
import torch, pandas as pd, librosa, jiwer
from opencc import OpenCC
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=rf"{ROOT}/model_for_finetune")
ap.add_argument("--adapter", default=None)
ap.add_argument("--corpus", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--tag", default="eval")
args = ap.parse_args()

cc = OpenCC("s2tw")


def norm(s):
    """中文 CER 前處理：NFKC、簡轉繁、去空白與標點、英文小寫。"""
    s = cc.convert(unicodedata.normalize("NFKC", str(s)))
    s = re.sub(r"[\s，。、？！：；「」『』（）《》〈〉…—\-,.?!:;\"'()\[\]~]", "", s)
    return s.lower()


proc = WhisperProcessor.from_pretrained(args.model)
model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model = model.to("cuda").eval()

corpus = Path(args.corpus)
df = pd.read_csv(corpus / f"{args.split}.tsv", sep="\t", quoting=3)
if args.limit:
    df = df.head(args.limit)

refs, hyps, rows = [], [], []
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
for i in range(0, len(df), args.bs):
    chunk = df.iloc[i:i + args.bs]
    wavs = [librosa.load(corpus / "clips" / p, sr=16000)[0] for p in chunk["path"]]
    feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feats.to("cuda", torch.bfloat16), language="zh", task="transcribe",
                             max_new_tokens=128)
    for (_, r), h in zip(chunk.iterrows(), proc.batch_decode(ids, skip_special_tokens=True)):
        ref, hyp = norm(r["sentence"]), norm(h)
        refs.append(ref); hyps.append(hyp)
        rows.append({"path": r["path"], "ref": r["sentence"], "hyp": h,
                     "cer": jiwer.cer(ref, hyp) if ref else None})
    print(f"  {min(i + args.bs, len(df))}/{len(df)}", flush=True)

elapsed = time.time() - t0
cer = jiwer.cer(refs, hyps) * 100
out = Path(rf"{ROOT}/results"); out.mkdir(exist_ok=True)
stem = f"{args.tag}_{corpus.name}_{args.split}"
pd.DataFrame(rows).to_csv(out / f"{stem}.tsv", sep="\t", index=False)
summary = {"tag": args.tag, "corpus": str(corpus), "split": args.split, "n": len(refs),
           "cer_pct": round(cer, 2), "sec_per_utt": round(elapsed / len(refs), 3),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "adapter": args.adapter}
json.dump(summary, open(out / f"{stem}.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)

for r in sorted(rows, key=lambda r: -(r["cer"] or 0))[:8]:
    print(f"[{(r['cer'] or 0) * 100:5.1f}%] 參考：{r['ref']}\n         辨識：{r['hyp']}")
print(json.dumps(summary, ensure_ascii=False))
