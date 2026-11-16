import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Шляхи будуємо від розташування скрипта, щоб він працював
# незалежно від того, з якої папки його запускають
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "arxiv_subset.parquet"
OUTPUT_DIR = ROOT / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"

MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64


def build_texts(df: pd.DataFrame) -> list[str]:
    """Об'єднує назву й анотацію у формат, на якому навчена модель: title [SEP] abstract."""
    titles = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str)
    return (titles + " [SEP] " + abstracts).tolist()


def main() -> None:
    # 1. Завантаження датасету
    df = pd.read_parquet(INPUT_FILE)
    print(f"Завантажено записів: {len(df)}")

    # 2. Підготовка текстів
    texts = build_texts(df)
    print(f"Приклад тексту:\n{texts[0][:200]}...\n")

    # 3. Завантаження моделі (під час першого запуску вона скачується з Hugging Face)
    model = SentenceTransformer(MODEL_NAME)

    # 4. Кодування текстів батчами, з прогресом і нормалізацією
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # 5. Перевірка результату
    print(f"\nОброблено текстів: {embeddings.shape[0]}")
    print(f"Розмірність ембедінгів: {embeddings.shape[1]}")
    print(f"Норма першого ембедінгу: {np.linalg.norm(embeddings[0]):.4f}")

    # 6–7. Збереження (спершу переконуємося, що папка існує)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings.astype("float32"))
    print(f"\nЗбережено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()