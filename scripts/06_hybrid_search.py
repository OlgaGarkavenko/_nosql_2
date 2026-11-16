import os
import math
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

SHOW = 5          # скільки результатів показуємо для кожного методу
RRF_K = 60        # стандартна константа RRF

QUERIES = [
    ("Точний термін", "BERT fine-tuning"),
    ("Ім'я автора", "Yann LeCun convolutional networks"),
    ("Перефразування", "making computers understand human emotions from text"),
]

# Часті слова, які не допомагають знайти документ (BM25 і так дає їм малу вагу,
# але без них результати чистіші)
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "we",
    "from", "with", "by", "at", "as", "this", "that", "be", "it", "its", "which",
}


# 1. BM25-індекс 

def tokenize(text: str) -> list[str]:
    """Нижній регістр, лише слова/числа (дефіс зберігаємо: fine-tuning), без стоп-слів."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(text).lower())
    return [w for w in words if w not in STOPWORDS]


def doc_text(row) -> str:
    """Текст документа для BM25: назва + анотація (назва двічі, щоб її слова важили більше)."""
    title = " ".join(str(row["title"]).split())
    abstract = " ".join(str(row["abstract"]).split())
    return f"{title} {title} {abstract}"


print("Будуємо BM25-індекс...")
corpus_tokens = [tokenize(doc_text(row)) for _, row in df.iterrows()]
bm25 = BM25Okapi(corpus_tokens)
corpus_sets = [set(toks) for toks in corpus_tokens]   # для швидкої перевірки збігів слів
print(f"BM25 готовий: {len(corpus_tokens)} документів")


# 4. Три функції пошуку

def bm25_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    """Повертає [(номер_рядка, BM25-оцінка), ...] лише для документів,
    де є хоча б одне слово запиту (решта для BM25 нерелевантні)."""
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    q_set = set(q_tokens)
    matched = [i for i, toks in enumerate(corpus_sets) if q_set & toks]
    matched.sort(key=lambda i: scores[i], reverse=True)
    return [(i, float(scores[i])) for i in matched[:top_k]]


def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    """Повертає [(номер_рядка, косинусна схожість), ...] з Pinecone."""
    q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    res = index.query(vector=q.tolist(), top_k=top_k, include_metadata=False)
    # id вигляду "paper_42" → 42 (номер рядка в df)
    return [(int(m.id.split("_")[1]), float(m.score)) for m in res.matches]


#  3. Reciprocal Rank Fusion

def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict]]:
    """Об'єднує кілька ранжованих списків.
    Бал документа = сума 1 / (k + rank) по всіх списках, де він є (rank з 1).
    Повертає [(номер_рядка, rrf_бал, {метод: місце}), ...] від найкращого."""
    scores: dict[int, float] = {}
    positions: dict[int, dict] = {}
    for method, docs in rankings.items():
        for rank, doc in enumerate(docs, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            positions.setdefault(doc, {})[method] = rank
    ordered = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [(d, scores[d], positions[d]) for d in ordered]


def hybrid_search(query: str, k: int = RRF_K, top_k: int = TOP_K) -> list[tuple[int, float, dict]]:
    """BM25 + векторний пошук, об'єднані через RRF."""
    rankings = {
        "BM25": [doc for doc, _ in bm25_search(query, top_k)],
        "vector": [doc for doc, _ in vector_search(query, top_k)],
    }
    return rrf(rankings, k=k)


# Вивід 

def title_of(i: int, n: int = 75) -> str:
    t = " ".join(str(df.loc[i, "title"]).split())
    return t if len(t) <= n else t[:n] + "..."


def info(i: int) -> str:
    return f"{df.loc[i, 'category']}, {int(df.loc[i, 'year'])}"


def print_simple(name: str, results: list[tuple[int, float]]) -> None:
    print(f"\n {name}")
    if not results:
        print("  Нічого не знайдено (жодного спільного слова з запитом).")
        return
    for rank, (i, score) in enumerate(results[:SHOW], start=1):
        print(f"  {rank}. [{score:.4f}] {title_of(i)} ({info(i)})")


def print_hybrid(results, bm25_top: set, vec_top: set, k: int) -> None:
    print(f"\n Гібридний (RRF, k={k}) ")
    for rank, (i, score, pos) in enumerate(results[:SHOW], start=1):
        where = ", ".join(f"{m} #{p}" for m, p in pos.items())
        new = "  ← немає в топ-5 жодного методу" if i not in bm25_top and i not in vec_top else ""
        print(f"  {rank}. [RRF {score:.5f}] {title_of(i)} ({info(i)})")
        print(f"     місця у списках: {where}{new}")


def main() -> None:
    for label, query in QUERIES:
        print(f"\n {label}: \"{query}\"")
        print(f"  токени BM25: {tokenize(query)}")

        # Рахуємо обидва списки один раз і використовуємо їх для всіх виводів
        # (hybrid_search робить те саме, але повторно звертався б до Pinecone)
        bm = bm25_search(query)
        vec = vector_search(query)
        print_simple("BM25", bm)
        print_simple("Векторний (Pinecone)", vec)

        bm25_top = {i for i, _ in bm[:SHOW]}
        vec_top = {i for i, _ in vec[:SHOW]}
        hyb = rrf({"BM25": [i for i, _ in bm], "vector": [i for i, _ in vec]}, k=RRF_K)
        print_hybrid(hyb, bm25_top, vec_top, RRF_K)

        # Порівняння k=60 і k=1 (для питання 3 у README)
        hyb_k1 = rrf({"BM25": [i for i, _ in bm], "vector": [i for i, _ in vec]}, k=1)
        top60 = [i for i, _, _ in hyb[:SHOW]]
        top1 = [i for i, _, _ in hyb_k1[:SHOW]]
        print(f"\n  k=60 vs k=1: топ-5 однаковий за складом: {set(top60) == set(top1)}, "
              f"за порядком: {top60 == top1}")
        if hyb and hyb_k1:
            r60 = hyb[0][1] / hyb[min(SHOW, len(hyb)) - 1][1]
            r1 = hyb_k1[0][1] / hyb_k1[min(SHOW, len(hyb_k1)) - 1][1]
            print(f"  відношення балу №1 до №{min(SHOW, len(hyb))}: k=60 → {r60:.2f}×, k=1 → {r1:.2f}×")

    # Короткий приклад формули RRF
    print("\nRRF-внесок за місце у списку, 1/(k+rank):")
    for k in (1, 60):
        row = ", ".join(f"#{r}: {1 / (k + r):.4f}" for r in (1, 2, 5, 10))
        print(f"  k={k:2d} → {row}  (перше місце важить у {(1 / (k + 1)) / (1 / (k + 10)):.1f}× більше за десяте)")


if __name__ == "__main__":
    main()