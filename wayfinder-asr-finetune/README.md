# wayfinder-asr-finetune

Taiwan Tongues ASR CE 的視障輔助短指令 LoRA 微調。

微調目標是降低路口噪音環境下短指令的字元錯誤率，例如「現在是紅燈嗎」「幫我找黑色的保溫杯」「停止」，
同時保留原模型對一般國語的辨識能力。

---

## 結果

| 測試集 | 說明 | 微調前 | 微調後 |
|:--|:--|--:|--:|
| 真人 40 句 | 錄音者未參與訓練；40 句中有 38 句的文字出現在訓練語料 | 4.51% | **0.00%** |
| 代理測試・乾淨 | 4 個訓練從沒出現過的合成聲音 | 3.07% | **0.31%** |
| 代理測試・加噪音 | 同上，再加訓練沒用過的 8 類環境噪音 | 4.71% | **0.31%** |
| 官方真人 200 句 | Taiwan Tongues 國語測試集，檢查一般國語能力有沒有退化 | 18.99% | **11.62%** |
| 官方・排除英文幻覺 | 扣掉原模型輸出整句英文的 4 句 | 12.97% | **11.08%** |

真人測試中，原模型把「現在是**紅**燈嗎」聽成「現在是**弘**燈嗎」，微調後修正；六組最小對立組（紅／綠燈、左／右、可以／不可以等）全部 0 錯。
逐句結果與最小對立組統計見 [`results/results.md`](results/results.md)。

CER 計算前以 NFKC、簡轉繁、去除空白與中英標點正規化，詳見 `eval_baseline.py`。

---

## 為什麼用 LoRA

基底模型為 whisper-large-v2 架構、15.5 億參數。官方全參數微調需求為 VRAM 40 GB 以上。
本倉庫以 LoRA 把可訓練參數壓到 786 萬（0.51%），單張 16 GB 消費級顯卡可跑。

| 項目 | 設定 |
|:--|:--|
| r／alpha／dropout | 16／32／0.05 |
| target modules | `q_proj`、`v_proj` |
| 精度 | 基底 bf16、LoRA 權重 fp32 |
| gradient checkpointing | 開（`use_reentrant=False`） |
| 每卡 batch × 梯度累積 | 8 × 2（等效 16） |
| learning rate | 1e-3，linear decay，warmup 50 步 |
| epochs | 2 |
| VRAM 峰值 | 約 5.0 GB |
| 訓練時間 | 55 分鐘（RTX 4080 16 GB） |

---

## 測試集錄製網站

<img src="docs/images/recorder_test_set.png" alt="測試集錄製網站" width="100%">

真人測試集用 `tools/recorder/` 的本機網頁錄製：一次顯示一句，空白鍵開始／停止，錄完自動檢查音量、爆音與長度，
直接存成 16 kHz 單聲道 WAV（`spkXX/NNN.wav`）與對應的 `spkXX.tsv`，可直接當 `test.tsv` 使用。

```bash
tools/recorder/start_recorder.bat
```

會在 `http://localhost:8765/recorder.html` 開啟（Chrome／Edge）。說話者代號用 `spk09` 這類代號，不要用真名或學號。
測試稿見 [`tools/recorder/錄音稿_測試集.md`](tools/recorder/錄音稿_測試集.md)。

---

## 語料

語料本身不公開，受試者同意範圍不涵蓋釋出。要收什麼、怎麼錄、怎麼標、怎麼切分，見 [`docs/語料需求規格.md`](docs/語料需求規格.md)。

指令涵蓋路口導引、尋物、物品登錄、文字識讀、空間記憶、系統控制與負例。
真人測試集 40 句，**錄音者完全不出現在訓練集**，不含噪音擴增。音檔規格：16 kHz、單聲道、WAV。

> 測試集採說話者留出，不隨機切分。隨機切會讓同一人的同一句同時出現在訓練與測試集。

---

## 重現

```bash
conda create -n ttasr python=3.11 -y
conda activate ttasr
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

取得基底模型（**要用可微調的 HuggingFace 格式；推論用的 CTranslate2 格式不能微調**）：

```bash
hf download adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0 --local-dir model_for_finetune
```

工作區結構（預設為倉庫根目錄，也可用環境變數 `TTASR_ROOT` 指定）：

```
model_for_finetune/          基底模型
sample_corpus/
└── wayfinder_cmd/           指令語料：train.tsv、test.tsv、clips/（依語料需求規格準備）
```

跑基準：

```bash
python eval_baseline.py --corpus sample_corpus/wayfinder_cmd --split test --tag baseline
```

微調：

```bash
python train_lora.py --datasets wayfinder_cmd --out out/lora-wayfinder
```

評估微調後：

```bash
python eval_baseline.py --corpus sample_corpus/wayfinder_cmd --split test --adapter out/lora-wayfinder --tag v1
```

直接使用本倉庫的 adapter 推論（自動從 Hugging Face 下載基底模型）：

```bash
python infer.py 你的錄音.wav
```

---

## 授權

Adapter 為 Taiwan Tongues ASR CE 的衍生作品，依 **TRAIL v0.1（TRAIL-M／TRAIL-S）** 釋出，全文見 [`LICENSE`](LICENSE)，
修改說明與使用限制見 [`NOTICE.md`](NOTICE.md)。再散布時須附上授權全文、修改說明與使用限制。

---

## 貢獻者設定

本倉庫 commit 不使用真實姓名：

```bash
git config user.name "wayfinder-ai"
```

```bash
git config user.email "<你的ID>+<你的帳號>@users.noreply.github.com"
```
