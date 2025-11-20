from typing import List, Dict, Any
import os

from openai import OpenAI

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.version_agent import VersionAgent


SYSTEM_PROMPT = """
你是一個工廠內部的「SOP 知識助理」。
你只能根據提供給你的 SOP 內容回答問題，不能自己亂編新的規定。

回答原則：
1. 優先使用最新版本的 SOP 內容。
2. 以條列式步驟回答，讓現場人員可以直接照做。
3. 必須在回答最後列出你參考到的 SOP 清單（包含 SOP_ID 與 VERSION）。
4. 若 SOP 沒有提到某件事，就明確說「SOP 中沒有相關說明」，不要猜。
"""

MODEL_NAME = "gpt-4.1-mini"  # 可以之後再改


class AnswerAgent:
    """
    回答代理：
    - 輸入：user_query + VersionAgent 篩過後的 evidences（最新 SOP 版本）
    - 輸出：一段自然語言回答（附 SOP 來源）
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("沒有找到環境變數 OPENAI_API_KEY，請先在 shell 裡設定。")

        self.client = OpenAI(api_key=api_key)
        self.model = MODEL_NAME

    def _build_context_text(self, evidences: List[Dict[str, Any]]) -> str:
        """
        把 evidences 轉成一大段「可放進 prompt 的文字」，
        讓模型知道每一份 SOP 的內容與版本。
        """
        blocks = []
        for ev in evidences:
            block = []
            block.append(f"SOP_ID: {ev.get('sop_id', '')}")
            block.append(f"VERSION: {ev.get('version', '')}")
            block.append(f"EFFECTIVE_DATE: {ev.get('effective_date', '')}")
            block.append(f"TITLE: {ev.get('title', '')}")
            block.append("CONTENT:")
            block.append(ev.get("text", "").strip())
            blocks.append("\n".join(block))

        return "\n\n" + ("-" * 80 + "\n\n").join(blocks)

    def run(self, user_query: str, evidences: List[Dict[str, Any]]) -> str:
        """
        回傳 LLM 生成的最終回答（string）。
        """
        if not evidences:
            # 沒有 evidence，就明確回覆查不到
            return "目前的 SOP 資料中找不到與你問題相關的內容，無法給出依據 SOP 的回答。"

        context_text = self._build_context_text(evidences)

        user_content = f"""
使用者的問題如下：
{user_query}

以下是與問題相關、且已經過版本篩選後的最新 SOP 內容（可能有多份）：
{context_text}

請你：
1. 僅根據上面提供的 SOP 內容回答。
2. 以條列式步驟或重點說明回答問題。
3. 清楚說明注意事項與安全相關提醒（如果 SOP 有提到）。
4. 在回答的最後，用「參考資料」區塊列出你用到的 SOP，例如：
   - SOP-001 v2.0 機台開機流程
   - SOP-003 v2.0 安全檢查流程
"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        answer = resp.choices[0].message.content.strip()
        return answer


# ---------- 整體串接 demo：RetrievalAgent + VersionAgent + AnswerAgent ----------

def _interactive_demo() -> None:
    retriever_agent = RetrievalAgent()
    version_agent = VersionAgent()
    answer_agent = AnswerAgent()

    print("SOP 多代理 demo：會依序執行 檢索 → 版本篩選 → 產生回答")
    print("輸入問題開始測試（輸入 exit 離開）。")

    while True:
        q = input("\n請輸入問題：").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # Step 1: 檢索多筆 evidence（含不同 SOP、不同版本）
        evidences = retriever_agent.run(q, top_k=8)

        if not evidences:
            print("沒有檢索到任何相關 SOP。")
            continue

        # Step 2: 只保留每個 SOP 的最新版本
        latest_evidences = version_agent.run(evidences)

        print(f"\n[Debug] 檢索到 {len(evidences)} 筆，版本篩選後剩 {len(latest_evidences)} 筆（每個 SOP 最新版）")

        # Step 3: 用最新 SOP 產生回答
        answer = answer_agent.run(q, latest_evidences)

        print("\n===== 最終回答 =====")
        print(answer)
        print("====================")


if __name__ == "__main__":
    _interactive_demo()
