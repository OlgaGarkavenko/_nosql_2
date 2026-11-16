# scripts/05_chunking.py
import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# Налаштування 
N_PAPERS = 30            # скільки статей з найдовшими анотаціями беремо
CHUNK_WORDS = 60         # розмір fixed-чанка (слів)
OVERLAP_WORDS = 10       # перекриття між сусідніми fixed-чанками (слів)
MAX_SEMANTIC_WORDS = 60  # максимум слів у semantic-чанку
BATCH_SIZE = 100
TOP_K = 5

INDEXES = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic",
}

# Запити підібрані під статті з фізики (перші 10 000 рядків arXiv).
# Якщо ваша вибірка з категорій cs.*, замініть їх на запити про ML.
TEST_QUERIES = [
    "dark matter distribution in galaxies",
    "quantum entanglement between particles",
    "numerical simulations of the early universe",
]

# Кінець речення: . ! ? після якого пробіл і велика літера, цифра або дужка
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


# 1. Вибір статей 

def clean(text: str) -> str:
    """Прибирає переноси рядків і зайві пробіли."""
    return " ".join(str(text).split())


def select_longest(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """Повертає n статей з найдовшими анотаціями (за кількістю слів)."""
    data = data.copy()
    data["abstract"] = data["abstract"].map(clean)
    data["title"] = data["title"].map(clean)
    data["n_words"] = data["abstract"].str.split().str.len()
    return data.sort_values("n_words", ascending=False).head(n)


# 2. Дві стратегії чанкування 

def fixed_size_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Ріже текст на шматки по `size` слів; сусідні шматки перекриваються на `overlap` слів."""
    if overlap >= size:
        raise ValueError("overlap має бути меншим за size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):   # дійшли до кінця тексту
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    """Ділить текст на речення за розділовими знаками."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def semantic_chunks(text: str, max_words: int = MAX_SEMANTIC_WORDS) -> list[str]:
    """Додає речення до чанка, поки не перевищимо max_words; речення ніколи не ріжуться."""
    chunks, current, current_len = [], [], 0
    for sentence in split_sentences(text):
        n = len(sentence.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def ends_mid_sentence(chunk: str) -> bool:
    """True, якщо чанк обривається не на кінці речення."""
    return not chunk.rstrip().endswith((".", "!", "?"))


# Підготовка записів 

def build_chunk_records(papers: pd.DataFrame, strategy: str) -> list[dict]:
    """Ріже кожну статтю на чанки і повертає список записів (поки без векторів)."""
    splitter = fixed_size_chunks if strategy == "fixed" else semantic_chunks
    records = []
    for row_idx, paper in papers.iterrows():
        chunks = splitter(paper["abstract"])
        for n, chunk in enumerate(chunks):
            records.append({
                "id": f"paper_{row_idx}_{strategy}_{n}",
                "text": chunk,
                "metadata": {
                    "arxiv_id": str(paper["id"]),
                    "title": paper["title"],
                    "chunk_text": chunk,
                    "chunk_index": n,
                    "total_chunks": len(chunks),
                    "year": int(paper["year"]),
                    "category": str(paper["category"]),
                    "strategy": strategy,
                },
            })
    return records


def print_stats(name: str, records: list[dict]) -> None:
    lengths = [len(r["text"].split()) for r in records]
    cut = sum(ends_mid_sentence(r["text"]) for r in records)
    print(f"  {name:9s}: чанків {len(records):4d} | слів у чанку: "
          f"мін {min(lengths)}, середнє {np.mean(lengths):.1f}, макс {max(lengths)} | "
          f"обриваються посеред речення: {cut} ({cut / len(records):.0%})")


def overlap_experiment(papers: pd.DataFrame) -> None:
    """Як overlap впливає на кількість чанків і на обсяг тексту, що зберігається (для README)."""
    total_words = papers["n_words"].sum()
    print(f"\nВплив overlap (size={CHUNK_WORDS} слів, усього слів в анотаціях: {total_words}):")
    for ov in (0, 10, 20, 30, 40):
        n_chunks, stored = 0, 0
        for text in papers["abstract"]:
            chunks = fixed_size_chunks(text, CHUNK_WORDS, ov)
            n_chunks += len(chunks)
            stored += sum(len(c.split()) for c in chunks)
        print(f"  overlap={ov:2d}: чанків {n_chunks:4d} | збережено слів {stored:5d} "
              f"({stored / total_words:.2f}× від оригіналу)")


# 3. Індекси Pinecone 

def get_index(name: str):
    """Створює індекс, якщо його немає, і повертає підключення до нього."""
    if not pc.has_index(name):
        print(f"Створюємо індекс '{name}'...")
        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)


# 4–5. Ембедінги і завантаження 

def embed_and_upload(index, records: list[dict], strategy: str) -> None:
    texts = [r["text"] for r in records]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Завантаження ({strategy})"):
        batch = [
            {"id": r["id"], "values": v.tolist(), "metadata": r["metadata"]}
            for r, v in zip(records[start:start + BATCH_SIZE], vectors[start:start + BATCH_SIZE])
        ]
        index.upsert(vectors=batch)

    # Статистика в Pinecone оновлюється із затримкою
    for _ in range(30):
        count = index.describe_index_stats().total_vector_count
        if count >= len(records):
            break
        time.sleep(2)
    print(f"Векторів в індексі ({strategy}): {count}")


# 6. Пошук по чанках 

def search_chunks(index, query: str, top_k: int = TOP_K):
    q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return index.query(vector=q.tolist(), top_k=top_k, include_metadata=True).matches


def print_chunk_results(strategy: str, matches) -> None:
    print(f"\n {strategy} ")
    if not matches:
        print("  Нічого не знайдено.")
        return
    for rank, m in enumerate(matches, start=1):
        meta = m.metadata
        text = meta["chunk_text"]
        snippet = text if len(text) <= 160 else text[:160] + "..."
        print(f"  {rank}. [{m.score:.4f}] {meta['title'][:70]}")
        print(f"     чанк {int(meta['chunk_index']) + 1}/{int(meta['total_chunks'])}: {snippet}")


def main() -> None:
    # 1. Вибираємо статті
    papers = select_longest(df, N_PAPERS)
    print(f"Обрано {len(papers)} статей; довжина анотацій від "
          f"{papers['n_words'].min()} до {papers['n_words'].max()} слів")

    # 2. Чанкування обома стратегіями
    records = {s: build_chunk_records(papers, s) for s in INDEXES}
    print("\nСтатистика чанків:")
    for s, recs in records.items():
        print_stats(s, recs)

    # Приклад чанків однієї статті для порівняння очима
    example_id = papers.index[0]
    print(f"\nПриклад: «{papers.loc[example_id, 'title'][:80]}»")
    for s, recs in records.items():
        print(f"\n  [{s}]")
        for r in recs:
            if r["id"].startswith(f"paper_{example_id}_"):
                print(f"   #{r['metadata']['chunk_index'] + 1}: {r['text']}")

    overlap_experiment(papers)

    # 3–5. Індекси, ембедінги, завантаження
    indexes = {}
    for s, name in INDEXES.items():
        print(f"\n=== {name} ===")
        indexes[s] = get_index(name)
        embed_and_upload(indexes[s], records[s], s)

    # 6. Пошук
    for query in TEST_QUERIES:
        print(f"\n Запит: \"{query}\"")
        for s, idx in indexes.items():
            print_chunk_results(s, search_chunks(idx, query))


if __name__ == "__main__":
    main()