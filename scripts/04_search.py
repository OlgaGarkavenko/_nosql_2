import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

EMBEDDINGS_FILE = "embeddings/embeddings.npy"
CURRENT_YEAR = datetime.now().year


# 2. Кодування запиту 

def encode_query(text: str) -> np.ndarray:
    """Перетворює текст запиту на нормалізований вектор (768 чисел).
    Нормалізуємо так само, як статті, інакше cosine ≠ dot product."""
    return model.encode(text, normalize_embeddings=True, convert_to_numpy=True)


# Допоміжні функції виводу 
def row_from_pinecone_id(vector_id: str) -> int:
    """'paper_42' -> 42: номер рядка в df і в embeddings.npy."""
    return int(vector_id.split("_")[1])


def short(text: str, n: int = 200) -> str:
    """Перші n символів тексту в один рядок."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def print_pinecone_results(title: str, response) -> list[str]:
    """Друкує результати Pinecone і повертає список їхніх id."""
    print(f"\n{title}")
    matches = response.matches
    if not matches:
        print("  Нічого не знайдено (жодна стаття не пройшла фільтр).")
        return []
    for rank, m in enumerate(matches, start=1):
        meta = m.metadata or {}
        row = df.iloc[row_from_pinecone_id(m.id)]
        print(f"{rank}. [{m.score:.4f}] {meta.get('title', row['title'])}")
        print(f"   категорія: {meta.get('category')} | рік: {int(meta.get('year', 0))} "
              f"| arXiv: {meta.get('arxiv_id')}")
        print(f"   {short(row['abstract'])}")
    return [m.id for m in matches]


# 3. Чистий семантичний пошук 

def semantic_search(query: str, top_k: int = TOP_K, flt: dict | None = None):
    """Пошук у Pinecone: вектор запиту + (необов'язковий) фільтр за метаданими."""
    return index.query(
        vector=encode_query(query).tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    )


# 5. Локальне порівняння метрик 

def local_top_k(query_vec: np.ndarray, emb: np.ndarray, k: int = TOP_K) -> dict:
    """Рахує три метрики для всіх статей і повертає топ-k номерів рядків для кожної."""
    dot = emb @ query_vec                                              # dot product
    cosine = dot / (np.linalg.norm(emb, axis=1) * np.linalg.norm(query_vec))  # cosine
    l2 = np.linalg.norm(emb - query_vec, axis=1)                       # L2-відстань

    return {
        # для схожості — чим більше, тим краще (сортуємо за спаданням)
        "cosine": (np.argsort(-cosine)[:k], cosine),
        "dot": (np.argsort(-dot)[:k], dot),
        # для відстані — чим менше, тим краще (сортуємо за зростанням)
        "l2": (np.argsort(l2)[:k], l2),
    }


def print_local(name: str, top_idx: np.ndarray, scores: np.ndarray) -> None:
    print(f"\n{name}")
    for rank, i in enumerate(top_idx, start=1):
        print(f"{rank}. [{scores[i]:.4f}] paper_{i} | {df.iloc[i]['category']} | "
              f"{short(df.iloc[i]['title'], 80)}")


def compare_metrics(query: str, emb: np.ndarray, title: str) -> dict:
    q = encode_query(query)
    res = local_top_k(q, emb)
    print(f"\n{title}")
    print_local("Cosine similarity (більше = схожіше)", *res["cosine"])
    print_local("Dot product (більше = схожіше)", *res["dot"])
    print_local("L2-distance (менше = схожіше)", *res["l2"])

    cos_ids, dot_ids, l2_ids = (list(res[m][0]) for m in ("cosine", "dot", "l2"))
    print("\nПорівняння порядку топ-5:")
    print(f"  cosine == dot: {cos_ids == dot_ids}")
    print(f"  cosine == L2:  {cos_ids == l2_ids}")
    return res


def main() -> None:
    query = "teaching machines to recognize objects in pictures"

    # 3. Чистий семантичний пошук 
    print(f"Запит: \"{query}\"")
    pinecone_ids = print_pinecone_results(
        "Семантичний пошук (без фільтрів)", semantic_search(query)
    )

    # 4. Пошук з фільтрацією 
    rl_query = "reinforcement learning"

    filter_a = {
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": CURRENT_YEAR - 5},
    }
    ids_a = print_pinecone_results(
        f"Приклад A: \"{rl_query}\", cs.LG, {CURRENT_YEAR - 5}–{CURRENT_YEAR}",
        semantic_search(rl_query, flt=filter_a),
    )

    filter_b = {"year": {"$lt": 2015}}
    ids_b = print_pinecone_results(
        f"Приклад B: \"{rl_query}\", будь-яка категорія, до 2015 року",
        semantic_search(rl_query, flt=filter_b),
    )

    print("\nПорівняння A і B:")
    print(f"  знайдено A: {len(ids_a)}, B: {len(ids_b)}, спільних: {len(set(ids_a) & set(ids_b))}")
    years = df["year"]
    print(f"  роки в датасеті: {years.min()}–{years.max()}; "
          f"статей cs.LG: {(df['category'] == 'cs.LG').sum()}")
    if not ids_a:
        print("  Приклад A порожній: у датасеті немає статей cs.LG за останні 5 років. "
              "Фільтр відсікає все ще ДО пошуку схожості.")

    # 5. Порівняння метрик на локальних ембедінгах 
    emb = np.load(EMBEDDINGS_FILE)
    print(f"\nЛокальні ембедінги: {emb.shape}, "
          f"норми від {np.linalg.norm(emb, axis=1).min():.4f} "
          f"до {np.linalg.norm(emb, axis=1).max():.4f}")

    res = compare_metrics(query, emb, "Нормалізовані ембедінги")

    # Звіряємо Pinecone з локальним cosine
    local_cos_ids = [f"paper_{i}" for i in res["cosine"][0]]
    print(f"\nPinecone топ-5 == локальний cosine топ-5: {pinecone_ids == local_cos_ids}")

    # Додатково (для README, питання 3): що буде без нормалізації.
    # Імітуємо різну довжину векторів, множачи кожен на випадкове число 0.5–2.0.
    rng = np.random.default_rng(42)
    unnormalized = emb * rng.uniform(0.5, 2.0, size=(len(emb), 1)).astype(emb.dtype)
    compare_metrics(query, unnormalized, "ДОДАТКОВО: НЕнормалізовані ембедінги (імітація)")


if __name__ == "__main__":
    main()