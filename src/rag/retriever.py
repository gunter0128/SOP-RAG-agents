import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from openai import OpenAI


# ---------- 路徑與常數設定 ----------

ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT_DIR / "data" / "index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = "text-embedding-3-small"


class Retriever:
    """
    簡單版 RAG 檢索器：
    - 載入 index_builder 產生的 embeddings.npy + metadata.json
    - 給一個 query，算 embedding
    - 用 cosine similarity 找出最相近的 SOP 文件
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("沒有找到環境變數 OPENAI_API_KEY，請先在 shell 裡設定。")

        self.client = OpenAI(api_key=api_key)

        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"找不到 embeddings 檔案：{EMBEDDINGS_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"找不到 metadata 檔案：{METADATA_PATH}")

        # 載入 embeddings 與 metadata
        self.embeddings = np.load(EMBEDDINGS_PATH)  # shape: (N, dim)

        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        if len(self.metadata) != self.embeddings.shape[0]:
            raise ValueError(
                f"metadata 筆數 ({len(self.metadata)}) 與 embeddings 筆數 "
                f"({self.embeddings.shape[0]}) 不一致"
            )

        # 預先做 L2 normalize，之後 cosine similarity 就是 dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings_normalized = self.embeddings / norms

    # ---------- 內部：query 轉 embedding ----------

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        # normalize
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec

    # ---------- 對外：搜尋函式 ----------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        給一個 query，回傳前 top_k 筆最相關的 SOP 片段：
        [
          {
            "sop_id": ...,
            "version": ...,
            "effective_date": ...,
            "title": ...,
            "text": ...,
            "file_name": ...,
            "score": 0.87
          },
          ...
        ]
        """
        if not query.strip():
            raise ValueError("query 不可為空字串。")

        q_vec = self._embed_query(query)  # shape: (dim,)

        # cosine similarity = dot( normalized_q, normalized_docs )
        scores = np.dot(self.embeddings_normalized, q_vec)  # shape: (N,)

        # 取分數最高的 top_k 筆
        top_k = min(top_k, scores.shape[0])
        idxs = np.argsort(-scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            doc = dict(self.metadata[idx])  # 複製一份
            doc["score"] = float(scores[idx])
            results.append(doc)

        return results


# ---------- 測試用：直接在終端試問問題 ----------

def _interactive_demo() -> None:
    print(f"載入 index 從：{INDEX_DIR}")
    retriever = Retriever()
    print("載入完成。輸入問題開始測試（輸入 exit 離開）。")

    while True:
        q = input("\n請輸入問題：").strip()
        if q.lower() in {"exit", "quit"}:
            break

        results = retriever.search(q, top_k=3)
        print(f"\n找到 {len(results)} 筆結果（只顯示前 3 筆）：")

        for i, r in enumerate(results, start=1):
            print("-" * 60)
            print(f"[{i}] {r['sop_id']} v{r['version']} ({r['effective_date']}) - {r['title']}")
            print(f"score = {r['score']:.4f}")
            print("內容片段：")
            print(r["text"])


if __name__ == "__main__":
    _interactive_demo()
