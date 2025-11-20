import sys
from pathlib import Path

# 自動把專案根目錄加入 Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import os

import streamlit as st

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.version_agent import VersionAgent
from src.agents.answer_agent import AnswerAgent


def init_agents():
    # 為了避免每次重跑，放在 session_state
    if "retrieval_agent" not in st.session_state:
        st.session_state["retrieval_agent"] = RetrievalAgent()
    if "version_agent" not in st.session_state:
        st.session_state["version_agent"] = VersionAgent()
    if "answer_agent" not in st.session_state:
        st.session_state["answer_agent"] = AnswerAgent()

    return (
        st.session_state["retrieval_agent"],
        st.session_state["version_agent"],
        st.session_state["answer_agent"],
    )


def main():
    st.set_page_config(page_title="SOP RAG Agents", page_icon="📘", layout="wide")

    st.title("SOP 多代理知識助理 Demo")
    st.write(
        "這個小系統會依序執行：**檢索 → 版本篩選 → 依據 SOP 生成回答**，"
        "幫助現場人員用自然語言查詢最新的作業流程。"
    )

    # 顯示 API key 設定狀態（避免沒設好一直報錯）
    api_key_ok = bool(os.environ.get("OPENAI_API_KEY"))
    with st.sidebar:
        st.header("環境狀態")
        if api_key_ok:
            st.success("OPENAI_API_KEY ✅ 已設定")
        else:
            st.error("OPENAI_API_KEY ❌ 尚未設定（請在 shell 裡用環境變數設定）")

        st.markdown("---")
        st.caption("提示：先在終端機中輸入：\n`$env:OPENAI_API_KEY = \"你的_API_KEY\"`")

    init_agents()
    retrieval_agent, version_agent, answer_agent = (
        st.session_state["retrieval_agent"],
        st.session_state["version_agent"],
        st.session_state["answer_agent"],
    )

    # 左右欄：左邊輸入問題＋回答，右邊 debug 顯示 evidence
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("問題輸入")

        st.markdown("""
        **你可以詢問以下類型的問題：**

        -  開機 / 關機流程  
        -  安全檢查  
        -  異常處理流程  
        -  品質檢驗步驟  
        -  SOP 版本與規範查詢  

        點下面的範例問題可以自動填入
        """)

        # ---- 範例問題（按鈕） ----
        example_questions = [
            "開機前需要做哪些安全檢查？",
            "設備發生異常時，現場人員應該怎麼處理？",
            "品質檢驗需要紀錄哪些項目？",
            "關機流程的步驟是什麼？",
            "最新版本的 SOP-005 說了哪些異常處理規定？",
        ]

        cols = st.columns(3)

        for i, q in enumerate(example_questions):
            col = cols[i % 3]
            if col.button(q):
                st.session_state["example_filled"] = q

        # ---- 主要輸入框 ----
        default_q = st.session_state.get("example_filled", "開機前需要做哪些安全檢查？")

        user_query = st.text_area(
            "請輸入你想查詢的問題：",
            value=default_q,
            height=120
        )


        top_k = st.slider("檢索筆數（含不同版本）", min_value=3, max_value=12, value=8, step=1)

        run_button = st.button("執行多代理查詢")

        if run_button:
            if not user_query.strip():
                st.warning("請先輸入問題。")
            elif not api_key_ok:
                st.error("尚未設定 OPENAI_API_KEY，請先在終端機設好再重新執行。")
            else:
                with st.spinner("正在檢索 SOP 並生成回答中..."):
                    # Step 1: 檢索
                    evidences = retrieval_agent.run(user_query, top_k=top_k)

                    if not evidences:
                        st.error("沒有檢索到任何相關 SOP。")
                    else:
                        # Step 2: 版本篩選
                        latest_evidences = version_agent.run(evidences)

                        # 暫存給右側 debug 顯示使用
                        st.session_state["last_evidences"] = evidences
                        st.session_state["last_latest_evidences"] = latest_evidences

                        # Step 3: 生成回答
                        answer = answer_agent.run(user_query, latest_evidences)

                        st.subheader("AI 回答（依據最新 SOP）")
                        st.markdown(answer)

    with col_right:
        st.subheader("檢索結果（debug 用）")

        evidences = st.session_state.get("last_evidences", [])
        latest_evidences = st.session_state.get("last_latest_evidences", [])

        if not evidences:
            st.info("還沒有檢索結果，請先在左側輸入問題並執行查詢。")
        else:
            with st.expander("原始檢索結果（可能含舊版 SOP）", expanded=False):
                for i, ev in enumerate(evidences, start=1):
                    st.markdown(
                        f"**[{i}] {ev['sop_id']} v{ev['version']} "
                        f"({ev['effective_date']})**  \n"
                        f"score = {ev['score']:.4f}  \n"
                        f"《{ev['title']}》"
                    )

            with st.expander("版本篩選後（每個 SOP 最新版）", expanded=True):
                for i, ev in enumerate(latest_evidences, start=1):
                    st.markdown(
                        f"**[{i}] {ev['sop_id']} v{ev['version']} "
                        f"({ev['effective_date']})**  \n"
                        f"《{ev['title']}》"
                    )
                    st.caption(ev["text"])


if __name__ == "__main__":
    main()
