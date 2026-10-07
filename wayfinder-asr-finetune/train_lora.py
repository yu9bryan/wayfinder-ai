"""Taiwan Tongues ASR CE（whisper-large-v2）LoRA 微調，16 GB 顯示卡版。

--datasets 是 sample_corpus/ 底下的資料集資料夾名稱，可用 + 串多份語料混訓；
--max_per_extra 可限制第二份之後每份的抽樣數。
"""
import os as _os
ROOT = _os.environ.get("TTASR_ROOT", ".")  # 工作區根目錄（預設為倉庫根目錄），見 README
import argparse, json, time
from dataclasses import dataclass
from pathlib import Path
import torch, pandas as pd, librosa
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=rf"{ROOT}/model_for_finetune")
ap.add_argument("--corpus_dir", default=rf"{ROOT}/sample_corpus")
ap.add_argument("--datasets", default="wayfinder_cmd")
ap.add_argument("--max_per_extra", type=int, default=0, help="第二份之後的語料每份最多取幾句，0=全取")
ap.add_argument("--out", default=rf"{ROOT}/out/lora-wayfinder")
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--targets", default="q_proj,v_proj")
ap.add_argument("--freeze_encoder", action="store_true")
ap.add_argument("--optim", default="adamw_torch")
ap.add_argument("--no_gc", action="store_true", help="關掉 gradient checkpointing：較快但吃記憶體")
args = ap.parse_args()

proc = WhisperProcessor.from_pretrained(args.model)
proc.tokenizer.set_prefix_tokens(language="zh", task="transcribe", predict_timestamps=False)


class TsvASR(Dataset):
    def __init__(self, items):
        self.items = items  # [(wav_path, sentence)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        wav, _ = librosa.load(path, sr=16000)
        feats = proc.feature_extractor(wav, sampling_rate=16000).input_features[0]
        labels = proc.tokenizer(text).input_ids
        return {"input_features": feats, "labels": labels}


def load_items(split):
    items = []
    for k, name in enumerate(args.datasets.split("+")):
        d = Path(args.corpus_dir) / name
        df = pd.read_csv(d / f"{split}.tsv", sep="\t", quoting=3)
        if k > 0 and args.max_per_extra and split == "train":
            df = df.sample(n=min(args.max_per_extra, len(df)), random_state=0)
        items += [(str(d / "clips" / p), s) for p, s in zip(df["path"], df["sentence"])]
        print(f"[data] {split}: {name} {len(df)} 句")
    return items


@dataclass
class Collator:
    def __call__(self, feats):
        x = proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats],
                                       return_tensors="pt")
        y = proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = y["input_ids"].masked_fill(y.attention_mask.ne(1), -100)
        # 模型會自己在最前面補 decoder_start_token，標籤裡的要拿掉
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": x["input_features"].to(torch.bfloat16), "labels": labels}


class VramLog(TrainerCallback):
    def on_log(self, a, state, control, logs=None, **kw):
        if logs is not None:
            logs["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)


model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
if args.freeze_encoder:
    model.model.encoder.requires_grad_(False)

lora = LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
                  target_modules=args.targets.split(","))
model = get_peft_model(model, lora)
for n, p in model.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()  # LoRA 權重用 fp32 存，前向用 bf16 autocast
model.print_trainable_parameters()

train_ds = TsvASR(load_items("train"))
targs = Seq2SeqTrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.accum,
    gradient_checkpointing=not args.no_gc,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # encoder 輸入不需要梯度，非 reentrant 才會算到 LoRA
    bf16=True,
    learning_rate=args.lr,
    warmup_steps=min(50, max(1, args.max_steps // 10)) if args.max_steps > 0 else 50,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    optim=args.optim,
    lr_scheduler_type="linear",
    logging_steps=5,
    save_strategy="no",
    report_to="tensorboard",
    logging_dir=str(Path(args.out) / "tb"),
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_num_workers=0,
)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                         data_collator=Collator(), callbacks=[VramLog()])
res = trainer.train()
model.save_pretrained(args.out)
summary = {"train_loss": res.training_loss, "steps": res.global_step, "minutes": round((time.time() - t0) / 60, 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "args": vars(args)}
json.dump(summary, open(Path(args.out) / "train_summary.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False))
