from typing import List, Dict, Any, Optional
from datetime import datetime

from src.agents.retrieval_agent import RetrievalAgent


def _parse_date(date_str: str) -> Optional[datetime]:
    """把 EFFECTIVE_DATE: YYYY-MM-DD 轉成 datetime，用來比較新舊。"""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _parse_version(version_str: str) -> Optional[float]:
    """把 '2.0' 類似字串轉成 float 方便比較，失敗就回傳 None。"""
    if not version_str:
        return None
    try:
        return float(version_str)
    except ValueError:
        return None


class VersionAgent:
    """
    版本代理：
    - 輸入：RetrievalAgent 回傳的 evidences list（可能包含 v1.0 / v2.0 / v3.0）
    - 輸出：每個 sop_id 僅保留「最新版本」的一筆 evidence
    """

    def run(self, evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        輸入：
            evidences: [
              {
                "sop_id": ...,
                "version": ...,
                "effective_date": ...,
                "title": ...,
                "text": ...,
                "file_name": ...,
                "score": ...
              },
              ...
            ]

        回傳：
            filtered_evidences: 只保留每個 sop_id 的最新版本，並依 score 降冪排序
        """
        if not evidences:
            return []

        best_by_sop: Dict[str, Dict[str, Any]] = {}

        for ev in evidences:
            sop_id = ev.get("sop_id", "")
            if not sop_id:
                # 資料不完整就略過
                continue

            current_best = best_by_sop.get(sop_id)

            if current_best is None:
                best_by_sop[sop_id] = ev
                continue

            # 比較哪一個比較「新」
            if self._is_newer(ev, current_best):
                best_by_sop[sop_id] = ev

        # 把結果取出並依 score 排序
        filtered = list(best_by_sop.values())
        filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return filtered

    def _is_newer(self, cand: Dict[str, Any], base: Dict[str, Any]) -> bool:
        """
        回傳 cand 是否比 base 新：
        比較順序：
        1. EFFECTIVE_DATE 比較（較晚的比較新）
        2. 若日期無法比較，再比較 VERSION 數字
        3. 若以上都無法比較，用 score 當 tie-breaker
        """
        # 1) 比日期
        cand_date = _parse_date(cand.get("effective_date", ""))
        base_date = _parse_date(base.get("effective_date", ""))

        if cand_date and base_date:
            if cand_date > base_date:
                return True
            if cand_date < base_date:
                return False
            # 日期一樣就往下比 version

        # 2) 比 version
        cand_ver = _parse_version(cand.get("version", ""))
        base_ver = _parse_version(base.get("version", ""))

        if cand_ver is not None and base_ver is not None:
            if cand_ver > base_ver:
                return True
            if cand_ver < base_ver:
                return False
            # 版本一樣就往下比 score

        # 3) 用 score 當最後 tie-breaker
        cand_score = cand.get("score", 0.0)
        base_score = base.get("score", 0.0)
        return cand_score > base_score


# ---------- 測試用：串 RetrievalAgent + VersionAgent ----------

def _interactive_demo() -> None:
    retriever_agent = RetrievalAgent()
    version_agent = VersionAgent()

    print("VersionAgent 測試模式：會先檢索，再只保留每個 SOP 最新版本。")
    print("輸入問題開始測試（輸入 exit 離開）。")

    while True:
        q = input("\n請輸入問題：").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # 先取多筆 evidence
        evidences = retriever_agent.run(q, top_k=8)
        print(f"\n檢索到 {len(evidences)} 筆（含不同版本）：")
        for i, ev in enumerate(evidences, start=1):
            print(f"  [{i}] {ev['sop_id']} v{ev['version']} ({ev['effective_date']}) score={ev['score']:.4f}")

        # 再用 VersionAgent 篩選成最新版本
        filtered = version_agent.run(evidences)
        print(f"\nVersionAgent 篩選後，保留每個 SOP 的最新版本，共 {len(filtered)} 筆：")

        for i, ev in enumerate(filtered, start=1):
            print("-" * 60)
            print(f"[{i}] {ev['sop_id']} v{ev['version']} ({ev['effective_date']}) - {ev['title']}")
            print(f"score = {ev['score']:.4f}")
            print("內容：")
            print(ev["text"])


if __name__ == "__main__":
    _interactive_demo()
