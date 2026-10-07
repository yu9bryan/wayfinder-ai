# NOTICE

本倉庫的 adapter 為 **Taiwan Tongues ASR CE** 的衍生作品（Derivative Work），依 TRAIL – TWM Responsible AI License v0.1 釋出。

## 原始作品

- 模型：`adi-gov-tw/Taiwan-Tongues-ASR-CE-pretrained-v1.0`
- 程式：<https://github.com/adi-gov-tw/Taiwan-Tongues-ASR-CE>
- 授權：TRAIL v0.1（全文見 `LICENSE`）

## 修改說明

1. 在原模型所有注意力層的 `q_proj`、`v_proj` 上加入 LoRA（r=16、alpha=32），以視障輔助短指令語料微調；原模型權重未變動。
2. 釋出內容為 LoRA adapter 權重（`adapter/adapter_model.safetensors`）與設定檔，需搭配原始基底模型使用。
3. 訓練與評估程式為本專案新撰寫，未修改原 repo 的 `train_asr.py`。

## 使用限制

使用本 adapter 及其衍生作品時，須遵守 TRAIL 附件 A 的使用限制（見 `LICENSE`），包括但不限於：
不得用於違法用途、傷害或歧視他人、未標示的機器生成內容散布、冒充人類，或對個人權利有不利影響的全自動決策。

再散布本 adapter 或其衍生作品時，須附上 `LICENSE`、本修改說明與上述使用限制。
