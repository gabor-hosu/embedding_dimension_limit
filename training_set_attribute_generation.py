import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import random
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")


def similarity(
    source_attributes: list[str],
    target_attributes: list[str],
    model: SentenceTransformer = model,
) -> float:
    source_embeddings = model.encode(
        source_attributes,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    if source_attributes is target_attributes:
        target_embeddings = source_embeddings
    else:
        target_embeddings = model.encode(
            target_attributes,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )

    similarities = model.similarity(target_embeddings, source_embeddings)
    similarities = similarities.cpu().numpy()

    if source_attributes is target_attributes:
        np.fill_diagonal(similarities, 0)

    return np.max(similarities, axis=-1).mean()


def average_similarity(
    source_attributes: list[str],
    target_attributes: list[str],
    num_of_samples: int = 3,
    model: SentenceTransformer = model,
):
    """Calculates stable semantic similarity by averaging multiple independent encoding runs."""

    if num_of_samples <= 0:
        raise ValueError("The sample number must be a positive integer!")

    accumulated_similarity = 0
    for _ in range(num_of_samples):
        accumulated_similarity += similarity(
            source_attributes=source_attributes,
            target_attributes=target_attributes,
            model=model,
        )

    return accumulated_similarity / num_of_samples


def filter_similar_words(
    source_attributes: list[str],
    target_attributes: list[str],
    similarity_threshold: float = 0.3,
    model: SentenceTransformer = model,
) -> list[str]:
    source_embeddings = model.encode(
        source_attributes,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    target_embeddings = model.encode(
        target_attributes,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    good_targets = []
    source_len = len(source_attributes)
    num_of_chunks = len(target_attributes) // source_len

    for i in range(num_of_chunks):
        start_idx = i * source_len
        end_idx = (i + 1) * source_len
        target_embs = target_embeddings[start_idx:end_idx]

        similarities = model.similarity(target_embs, source_embeddings)
        similarities = similarities.cpu().numpy()

        rows, cols = np.where(similarities > similarity_threshold)
        bad_indexes = set(rows)
        good_indexes = set(range(len(target_embs))) - bad_indexes

        good_targets.extend([target_attributes[start_idx + j] for j in good_indexes])

    remainder_start = num_of_chunks * source_len
    if remainder_start < len(target_attributes):
        target_embs = target_embeddings[remainder_start:]
        similarities = model.similarity(target_embs, source_embeddings)
        similarities = similarities.cpu().numpy()

        rows, cols = np.where(similarities > similarity_threshold)
        bad_indexes = set(rows)
        good_indexes = set(range(len(target_embs))) - bad_indexes

        good_targets.extend(
            [target_attributes[remainder_start + j] for j in good_indexes]
        )

    return good_targets


def main():
    with open("./dataset/limit_dataset/target_attributes.txt") as file:
        target_attributes = [line.strip() for line in file]

    df_train = pd.read_csv("./dataset/limit_dataset/generated_attributes.csv")
    source_attributes = df_train["liked_item"].to_list()

    filtered_attributes = filter_similar_words(
        source_attributes,
        target_attributes,
    )

    cross_similarity = average_similarity(source_attributes, filtered_attributes)
    target_self_similarity = average_similarity(
        filtered_attributes, filtered_attributes
    )
    source_self_similarity = average_similarity(source_attributes, source_attributes)

    print(f"len(filtered_attributes) = {len(filtered_attributes)}")
    print(f"Similarity target_attributes with source_attributes: {cross_similarity}")
    print(f"Inner similarity of target_attributes: {target_self_similarity}")
    print(f"Inner similarity of source_attributes: {source_self_similarity}")

    answer = input("Do you want to save the current results? (y/n)\n")
    if answer in {"y", "Y"}:
        with open(
            "./dataset/limit_dataset/filtered_target_attributes.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write("\n".join(filtered_attributes) + "\n")
        print(
            "Results saved in ./dataset/limit_dataset/filtered_target_attributes.txt."
        )


if __name__ == "__main__":
    main()
    # Results:
    # len(filtered_attributes) = 1857
    # Similarity target_attributes with source_attributes: 0.26390916109085083
    # Inner similarity of target_attributes: 0.6524001359939575
    # Inner similarity of source_attributes: 0.6683806777000427
