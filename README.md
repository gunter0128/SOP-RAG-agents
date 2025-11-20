#  SOP RAG Agents

### SOP 文件 × 版本管理 × 作業流程查詢 × AI 多代理助理

> 使用「自動分段與編索引的 SOP 文件（多版本）＋語意檢索（Embedding）＋LLM 多代理（Agents）」
> 自動化產生 **作業流程回答、指示步驟、引用最新版本 SOP**，支援製造/設備/生產部門日常 SOP 查詢與決策。

---

# 1. 為什麼做這個主題？

### 這是符合AI engineer職位跟我的研究相契合的實際體現

SOP 是所有製造現場的重要依據：
**開機、關機、安全檢查、異常處理、品質檢驗** 全都靠 SOP 控制。

本專案展示：

* 能處理**多版本 SOP 文件**
* 能做 **Embedding 索引（index builder）**
* 能用 **Retriever → Version Selector → LLM Answering** 的 RAG pipeline
* **三個 AI Agents 分工合作**
* **可以完全 Demo 的 Streamlit dashboard**

專案目的：
**讓主管看到我能把資料（SOP）→ 索引 → 模型 → 多代理 → UI 整成一套真正能用的企業 AI 系統。**

---
## 專案快速總覽（Problem / Input / Output）
### 要解決的問題（Problem）

現場工程師在查 SOP 時常遇到：

* 文件太長、不易搜尋

* 不確定哪一份才是最新版本

* 急需步驟時必須手動翻找內容

* 造成操作不一致、時間浪費、甚至用到舊 SOP

本專案透過 RAG + 多代理，協助工程師用自然語言快速取得「最新版本 SOP」的正確流程。

## 系統輸入（Input）

### 多版本 SOP Markdown 文件（模擬自建資料集）

* SOP_ID

* VERSION

* EFFECTIVE_DATE

* TITLE

### 步驟內容（純文字）

使用者自然語言問題
例如：

「開機前需要做哪些安全檢查？」

「設備發生異常時要怎麼處理？」

「關機流程有哪些步驟？」

## 系統輸出（Output）

* 條列式、可直接執行的 SOP 步驟

* 僅根據最新版本 SOP

*回答最後列出引用來源，例如：

參考資料：
- SOP-001 v2.0 機台開機流程
- SOP-003 v2.0 安全檢查流程
---

# 2. 系統架構概觀

整體分成四層：

---

## 2.1 文件處理層（Document Prep）

來源資料：
`data/sop_raw/*.md`（多版本）

工作：

* 解析 SOP metadata（SOP_ID / Version / Effective Date）
* 分段 (chunking)
* 建立 `metadata.json`

---

## 2.2 Indexing 層（Embedding Search Index）

使用模型：

```
text-embedding-3-small
```

產出：

* `embeddings.npy`（所有 SOP chunk 的向量）
* `metadata.json`（每段對應的 SOP 資訊）

用途：
後續使用 cosine similarity 做語意檢索。

---

## 2.3 AI Agents 層（RAG Multi-Agents）

本專案使用 **三個 Agents**：

---

### **Agent 1 — Retrieval Agent**

* 將使用者的自然語言詢問轉成 embedding
* 從所有 SOP 中找出最相關內容（含不同版本）

---

### **Agent 2 — Version Agent**

SOP 常有版本：
`v1.0 / v1.1 / v2.0 / v3.0 ...`

這個 Agent 用三步驟挑出「真正最新」的版本：

1. **Effective Date（生效日期）**
2. **Version number**
3. **Similarity score（tie-breaker）**

輸出：
每個 SOP 只保留「最新版本」。

---

### **Agent 3 — Answer Agent**

* 使用最新版本 SOP 不會亂編答案
* 條列式產生操作步驟
* 回答底部列出參考 SOP（SOP_ID + Version）

---

## 2.4 Streamlit DEMO 層（UI）

功能：

✔ 左側：輸入問題（可選範例問題）

✔ 中央：AI 回答（依據 SOP）

✔ 右側：檢索結果 & 版本篩選結果（Debug 用）


---

# 3. 使用資料集：多版本 SOP 資料集（自行建立）

本專案使用自行建立的 SOP ：

```
SOP-001_startup_v1.0.md
SOP-001_startup_v2.0.md
SOP-002_shutdown_v1.0.md
...
SOP-005_error_handling_v3.0.md
```

內容包含：

* 開機流程
* 關機流程
* 安全檢查流程
* 品質檢查
* 異常處理流程

格式：`.md`

放在：

```
data/sop_raw/
```

---

# 4. 專案目錄結構

```
sop-rag-agents/
├── data/
│   ├── sop_raw/                   # 原始多版本 SOP
│   └── index/                     # embeddings + metadata（自動生成）
│
├── src/
│   ├── rag/
│   │   ├── index_builder.py       # 構建 Embedding 索引
│   │   └── retriever.py           # 語意檢索
│   │
│   ├── agents/
│   │   ├── retrieval_agent.py     # Agent 1：檢索
│   │   ├── version_agent.py       # Agent 2：版本管理
│   │   └── answer_agent.py        # Agent 3：LLM 回答
│   │
│   └── app/
│       └── dashboard.py           # Streamlit Dashboard（Demo）
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 5. 如何重現專案

## 5.1 建立環境

```bash
git clone https://github.com/<your-account>/sop-rag-agents.git
cd sop-rag-agents

python -m venv .venv
.\.venv\Scripts\activate     # Windows

pip install -r requirements.txt
$env:OPENAI_API_KEY="你的API_KEY"
```

---

## 5.2 建立 SOP 索引

```bash
python -m src.rag.index_builder
```

會產生：

```
data/index/embeddings.npy
data/index/metadata.json
```

---

## 5.3 執行 Streamlit Demo（重點）

```bash
streamlit run src/app/dashboard.py
```

功能包含：

* AI 互動查詢 SOP
* 多版本 SOP 智慧選擇
* 自動產生操作步驟
* 自動列出參考 SOP

---

# 6. RAG 與版本管理的可解釋性設計

## 6.1 語意檢索（Retriever）

使用 cosine similarity 做排序
embedding 來源：`text-embedding-3-small`

輸出會包含：

* SOP_ID
* VERSION
* EFFECTIVE_DATE
* TEXT content
* similarity score

---

## 6.2 版本管理（Version Agent）

比對方式：

1. **日期較新優先**
2. **版本號較大優先**
3. **必要時以 score_break 再比較**

確保 LLM 只會看到「最新規範」，避免引用舊 SOP。

---

## 6.3 LLM 回答設計

LLM 嚴格根據 SOP：

* 條列式步驟
* 注意事項
* 明確說明引用了哪一份 SOP（含版本號）
* SOP 未提到的禁止亂回答（避免 hallucination）

---

# 7. 企業價值（Business Impact）

### **① SOP 查詢自動化**

現場工程師不需翻文件即可詢問：

* 開機流程
* 關機流程
* 安全檢查
* 異常處理

即時取得標準流程。

---

### **② 確保使用“最新版本 SOP”**

避免：

* 舊版 SOP 被誤用
* 生產流程出錯
* QA / Audit 問題

---

### **③ 教育訓練輔助**

新人不熟 SOP → 直接問
回覆是最新版本 SOP，非常可靠。

---

### **④ 可整合到 TSMC 內部系統**

* 可加入 MES（製造執行系統）
* 可接設備異常紀錄 → 自動查 SOP
* 可在內部 Portal / Chatbot 提供查詢

---

### **⑤ 可延伸到整個製造流程（SCM / Equipment / Quality）**

只要換資料，就能支援：

* 機台維護手冊
* 設備異常碼（Error code）查詢
* QC 品質規範
* 安全流程
