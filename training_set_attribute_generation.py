import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import random
import numpy as np

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")


def down_sample(
    source_attributes: list[str],
    target_attributes: list[str],
) -> tuple[list[str], list[str]]:
    if len(target_attributes) < len(source_attributes):
        source_attributes = random.sample(source_attributes, len(target_attributes))
        return source_attributes, target_attributes

    target_attributes = random.sample(target_attributes, len(source_attributes))
    return source_attributes, target_attributes


def similarity(
    source_attributes: list[str],
    target_attributes: list[str],
    model: SentenceTransformer = model,
) -> float:
    if len(source_attributes) != len(target_attributes):
        raise ValueError("The source and target attributes length must be the same!")

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


def filter_similar_words(
    source_attributes: list[str],
    target_attributes: list[str],
    similarity_threshold: float = 0.6,
    model: SentenceTransformer = model,
) -> list[str]:
    if len(source_attributes) != len(target_attributes):
        raise ValueError("The source and target attributes length must be the same!")

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

    similarities = model.similarity(target_embeddings, source_embeddings)
    similarities = similarities.cpu().numpy()

    rows, cols = np.where(similarities > similarity_threshold)
    rows = np.unique(rows)

    bad_indexes = set(rows)
    all_indexes = set(list(range(len(target_attributes))))
    good_indexes = all_indexes - bad_indexes
    return [target_attributes[i] for i in good_indexes]


def main():
    with open("./dataset/limit_dataset/target_attributes.txt") as file:
        target_attributes = [line.strip() for line in file]

    df_train = pd.read_csv("./dataset/limit_dataset/generated_attributes.csv")
    source_attributes = df_train["liked_item"].to_list()

    source_attributes, target_attributes = down_sample(
        source_attributes, target_attributes
    )

    filtered_attributes = filter_similar_words(
        source_attributes,
        target_attributes,
    )

    source_attributes, filtered_attributes = down_sample(
        source_attributes, filtered_attributes
    )

    cross_similarity = similarity(source_attributes, filtered_attributes)
    target_self_similarity = similarity(filtered_attributes, filtered_attributes)
    source_self_similarity = similarity(source_attributes, source_attributes)

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
