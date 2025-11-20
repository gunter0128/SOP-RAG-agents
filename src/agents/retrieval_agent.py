from typing import List, Dict, Any

from src.rag.retriever import Retriever


class RetrievalAgent:
    """
    檢索代理：
    - 封裝 Retriever
    - 對外只提供 run(query, top_k) 介面
    - 回傳「已排序的 evidence 清單」
    """

    def __init__(self) -> None:
        # 可以在這裡共用同一個 Retriever 實例
        self.retriever = Retriever()

    def run(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        給使用者問題，回傳 evidence list：
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
        results = self.retriever.search(query=query, top_k=top_k)

        # 這裡可以再做一次排序（理論上 retriever 已經排過了）
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        return results_sorted


# ---------- 測試用：直接在終端試問問題 ----------

def _interactive_demo() -> None:
    agent = RetrievalAgent()
    print("RetrievalAgent 已啟動，輸入問題開始測試（輸入 exit 離開）。")

    while True:
        q = input("\n請輸入問題：").strip()
        if q.lower() in {"exit", "quit"}:
            break

        evidences = agent.run(q, top_k=5)
        print(f"\n前 {len(evidences)} 筆 evidence：")

        for i, ev in enumerate(evidences, start=1):
            print("-" * 60)
            print(f"[{i}] {ev['sop_id']} v{ev['version']} ({ev['effective_date']}) - {ev['title']}")
            print(f"score = {ev['score']:.4f}")
            print("內容：")
            print(ev["text"])


if __name__ == "__main__":
    _interactive_demo()
