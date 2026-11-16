import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс 
if not pc.has_index(INDEX_NAME):
    print(f"Створюємо індекс '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",   # ембедінги нормалізовані, тому cosine = dot product
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Чекаємо, поки індекс стане готовим до роботи
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
    print("Індекс створено.")
else:
    print(f"Індекс '{INDEX_NAME}' уже існує.")

# Підключаємося до індексу
index = pc.Index(INDEX_NAME)

# Завантажуємо дані
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)
print(f"Статей: {len(df)}, ембедінгів: {embeddings.shape}")

# Перевірки: кількість рядків і розмірність мають збігатися
if len(df) != len(embeddings):
    raise SystemExit("Кількість статей і ембедінгів різна. Перезапустіть 02_embed.py")
if embeddings.shape[1] != VECTOR_DIM:
    raise SystemExit(f"Розмірність ембедінгів {embeddings.shape[1]}, а очікується {VECTOR_DIM}")


def text_or_empty(value) -> str:
    """Перетворює значення на рядок; порожні значення (None, NaN) стають ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def build_batch(start: int, end: int) -> list[dict]:
    """Формує список записів для Pinecone з рядків start..end-1."""
    batch = []
    for i in range(start, end):
        row = df.iloc[i]
        batch.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": text_or_empty(row["id"]),
                "title": text_or_empty(row["title"]),
                "abstract": text_or_empty(row["abstract"])[:500],
                "authors": text_or_empty(row["authors"])[:200],
                "year": int(row["year"]),
                "category": text_or_empty(row["category"]),
            },
        })
    return batch


def main() -> None:
    # Завантажуємо батчами з прогресом
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження в Pinecone"):
        end = min(start + BATCH_SIZE, len(df))
        index.upsert(vectors=build_batch(start, end))

    # Статистика оновлюється із затримкою, тому трохи чекаємо
    for _ in range(30):
        stats = index.describe_index_stats()
        if stats.total_vector_count >= len(df):
            break
        time.sleep(2)

    print(f"\nВекторів в індексі '{INDEX_NAME}': {stats.total_vector_count}")


if __name__ == "__main__":
    main()