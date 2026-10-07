# Model Card：Wayfinder 短指令 ASR Adapter

## 概述

Taiwan Tongues ASR CE 的 LoRA adapter，針對視障輔助應用的**短指令辨識**微調。

使用情境為戶外路口與室內噪音環境下的單句指令，典型長度 1–5 秒。

---

## 基底模型

| 項目 | 內容 |
|:--|:--|
| 名稱 | Taiwan Tongues ASR CE（`adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0`） |
| 發布單位 | 數位發展部數位產業署 |
| 架構 | whisper-large-v2 |
| 參數量 | 15.5 億 |
| 支援語言 | 國語、台語、客語、英語 |
| 授權 | TRAIL v0.1（TWM Responsible AI License） |

---

## 微調設定

| 項目 | 數值 |
|:--|:--|
| 方法 | LoRA（PEFT 0.21.2） |
| r／alpha／dropout | 16／32／0.05 |
| target modules | `q_proj`、`v_proj`（共 96 個注意力模組：encoder 自注意力 32、decoder 自注意力 32、cross-attention 32） |
| 可訓練參數 | 7,864,320（佔總參數 0.51%） |
| learning rate | 1e-3，linear decay，warmup 50 步 |
| epochs | 2 |
| batch × grad accum | 8 × 2（等效 16） |
| 精度 | 基底 bf16、LoRA 權重 fp32、bf16 autocast |
| gradient checkpointing | 開，`use_reentrant=False`（encoder 輸入不需要梯度，非 reentrant 才算得到 encoder 端 LoRA 的梯度） |
| 語系 token | `<|zh|>`、`<|transcribe|>`、`<|notimestamps|>` |
| 訓練硬體 | NVIDIA RTX 4080 16 GB，VRAM 峰值約 5.0 GB |
| 訓練時間 | 55 分鐘 |

---

## 評估資料

| 集合 | 句數 | 說話者 | 說明 |
|:--|--:|:--|:--|
| 真人測試 | 40 | 1 人 | 真人錄音，錄音者不在訓練集，無擴增 |
| 官方測試 | 200 | 137 人 | Taiwan Tongues 國語測試集，說話者不在訓練集 |

指令涵蓋類別：路口導引、尋物、物品登錄、文字識讀、空間記憶、系統控制、負例，並刻意讓六組最小對立組成對出現
（紅燈／綠燈、左邊／右邊、前面／後面、可以／不可以、現在／等一下、開始／停止）。

訓練語料不公開，受試者同意範圍不涵蓋釋出。語料要求見 [`docs/語料需求規格.md`](docs/語料需求規格.md)。

---

## 評估

| 測試集 | 說明 | 微調前 | 微調後 |
|:--|:--|--:|--:|
| 真人 40 句 | 錄音者未參與訓練；40 句中有 38 句的文字出現在訓練語料 | 4.51% | **0.00%** |
| 代理測試・乾淨 | 4 個訓練從沒出現過的合成聲音 | 3.07% | **0.31%** |
| 代理測試・加噪音 | 同上，再加訓練沒用過的 8 類環境噪音 | 4.71% | **0.31%** |
| 官方真人 200 句 | Taiwan Tongues 國語測試集，檢查一般國語能力有沒有退化 | 18.99% | **11.62%** |
| 官方・排除英文幻覺 | 扣掉原模型輸出整句英文的 4 句 | 12.97% | **11.08%** |

正規化規則：NFKC → OpenCC `s2tw` → 去除空白與中英標點 → 英文小寫（`eval_baseline.py` 的 `norm()`）。

最小對立組專項檢查（真人 40 句）：

| 對立組 | 微調前錯誤數 | 微調後錯誤數 |
|:--|--:|--:|
| 紅燈／綠燈 | 1（紅燈 → 弘燈） | 0 |
| 左／右 | 0 | 0 |
| 可以／不可以 | 0 | 0 |
| 前面／後面 | 0 | 0 |
| 開始／停止 | 0 | 0 |
| 現在／等一下 | 0 | 0 |

詳細結果見 [`results/results.md`](results/results.md)。

---

## 使用限制

- **辨識結果不得單獨作為通行決策依據。**
- 使用與再散布須遵守 TRAIL v0.1 附件 A 的使用限制（見 `LICENSE`）。

---

## 授權

TRAIL v0.1。Adapter 為基底模型的衍生作品，依 TRAIL-M 釋出；程式碼依 TRAIL-S 釋出。修改說明見 `NOTICE.md`。
