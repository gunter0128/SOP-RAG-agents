import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from openai import OpenAI


# ---------- 路徑設定 ----------

# index_builder.py 在 src/rag 底下 → 往上兩層就是專案根目錄
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "sop_raw"
INDEX_DIR = ROOT_DIR / "data" / "index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = "text-embedding-3-small"


# ---------- 讀取與解析 SOP 檔案 ----------

def parse_sop_file(path: Path) -> Dict[str, Any]:
    """
    解析單一 SOP .md 檔案，回傳：
    {
        "sop_id": ...,
        "version": ...,
        "effective_date": ...,
        "title": ...,
        "text": ...,
        "file_name": ...
    }
    """
    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    header = {}
    body_lines: List[str] = []

    for i, line in enumerate(lines):
        if line.startswith("SOP_ID:"):
            header["sop_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("VERSION:"):
            header["version"] = line.split(":", 1)[1].strip()
        elif line.startswith("EFFECTIVE_DATE:"):
            header["effective_date"] = line.split(":", 1)[1].strip()
        elif line.startswith("TITLE:"):
            header["title"] = line.split(":", 1)[1].strip()
            body_lines = lines[i + 1 :]
            break

    body_text = "\n".join(body_lines)

    record = {
        "sop_id": header.get("sop_id", ""),
        "version": header.get("version", ""),
        "effective_date": header.get("effective_date", ""),
        "title": header.get("title", ""),
        "text": body_text,
        "file_name": path.name,
    }

    return record


def load_all_sop_documents() -> List[Dict[str, Any]]:
    """
    掃描 data/sop_raw/ 底下所有 .md 檔案並解析
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"sop_raw 目錄不存在：{RAW_DIR}")

    docs: List[Dict[str, Any]] = []

    for path in sorted(RAW_DIR.glob("*.md")):
        doc = parse_sop_file(path)
        docs.append(doc)

    if not docs:
        raise RuntimeError(f"在 {RAW_DIR} 沒有找到任何 .md SOP 檔案")

    return docs


# ---------- 建立 Embeddings ----------

def build_embeddings(docs: List[Dict[str, Any]]) -> np.ndarray:
    """
    對每一份 SOP 文件的 text 建立 embedding，回傳 numpy array
    shape: (num_docs, dim)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("沒有找到環境變數 OPENAI_API_KEY，請先在 shell 裡設定。")

    client = OpenAI(api_key=api_key)

    texts = [d["text"] for d in docs]

    # OpenAI embeddings API 一次可以丟多筆 input
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    vectors = [item.embedding for item in resp.data]
    embeddings = np.array(vectors, dtype="float32")

    return embeddings


# ---------- 儲存 index ----------

def save_index(embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
    """
    將 embeddings 與 metadata 存到 data/index/ 底下
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # numpy 向量
    np.save(EMBEDDINGS_PATH, embeddings)

    # metadata 存 json
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------- main ----------

def main() -> None:
    print(f"專案根目錄：{ROOT_DIR}")
    print(f"讀取 SOP 來源：{RAW_DIR}")

    docs = load_all_sop_documents()
    print(f"找到 {len(docs)} 份 SOP 文件，開始建立 embeddings...")

    embeddings = build_embeddings(docs)
    print(f"embeddings shape = {embeddings.shape}")

    save_index(embeddings, docs)
    print(f"已儲存 embeddings 至：{EMBEDDINGS_PATH}")
    print(f"已儲存 metadata 至：{METADATA_PATH}")
    print("完成建立 SOP 索引。")


if __name__ == "__main__":
    main()
