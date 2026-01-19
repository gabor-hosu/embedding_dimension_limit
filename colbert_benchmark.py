import os

os.environ["TQDM_DISABLE"] = "1"

from pylate import indexes, models, retrieve
from datasets import load_dataset
import pandas as pd


def store_docs(
    corpus: pd.DataFrame,
    model: models.ColBERT,
    index_folder: str = "./dataset/limit_small_index",
    index_name: str = "limit_small_index",
):
    index = indexes.Voyager(
        index_folder=index_folder,
        index_name=index_name,
        override=True,
    )

    documents_ids = corpus["_id"].to_list()
    documents = corpus["text"].to_list()

    documents_embeddings = model.encode(
        documents,
        batch_size=32,
        is_query=False,
        show_progress_bar=True,
    )

    index.add_documents(
        documents_ids=documents_ids,
        documents_embeddings=documents_embeddings,
    )


def retrieve_docs(
    top_k: int,
    queries: pd.DataFrame,
    model: models.ColBERT,
    index_dir: str,
    show_progress: bool = False,
    index_name: str = "limit_small_index",
):
    index = indexes.Voyager(
        index_folder=index_dir,
        index_name=index_name,
    )

    retriever = retrieve.ColBERT(index=index)

    queries_embeddings = model.encode(
        queries["text"].to_list(),
        batch_size=32,
        is_query=True,
        show_progress_bar=show_progress,
    )

    retrieved_docs = dict()
    for idx, query_emb in enumerate(queries_embeddings):
        results = retriever.retrieve(
            queries_embeddings=query_emb, k=top_k, device="cuda"
        )
        retrieved_docs[f"query_{idx}"] = {result["id"] for result in results[0]}

    return retrieved_docs


def benchmark(
    recall_at: int,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    index_dir: str,
    model: models.ColBERT,
) -> float:
    retrieved_docs = retrieve_docs(
        top_k=recall_at, queries=queries, model=model, index_dir=index_dir
    )
    relevant_docs = qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()

    total = 0
    for qid in queries["_id"]:
        total += len(retrieved_docs[qid] & relevant_docs[qid]) / len(relevant_docs[qid])

    return total / len(relevant_docs)


def benchmark_on(
    index_dir: str,
    top_ks: list[int],
    dataset_name: str = None,
    dataset_dir: str = None,
    is_first_run: bool = False,
):
    if not dataset_name and not dataset_dir:
        print("Warning: No dataset provided!")
        return

    if dataset_dir:
        qrels = load_dataset(
            "json", data_files=f"{dataset_dir}/qrels.jsonl", split="all"
        ).to_pandas()
        corpus = load_dataset(
            "json", data_files=f"{dataset_dir}/corpus.jsonl", split="all"
        ).to_pandas()
        queries = load_dataset(
            "json", data_files=f"{dataset_dir}/queries.jsonl", split="all"
        ).to_pandas()
    else:
        qrels = load_dataset(dataset_name, "default", split="all").to_pandas()
        corpus = load_dataset(dataset_name, "corpus", split="all").to_pandas()
        queries = load_dataset(dataset_name, "queries", split="all").to_pandas()

    model = models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cuda",
    )

    if is_first_run:
        store_docs(corpus=corpus, model=model, index_folder=index_dir)

    for top_k in top_ks:
        print(
            f"Recall@{top_k} = {benchmark(recall_at=top_k, queries=queries, qrels=qrels, model=model, index_dir=index_dir)}"
        )


if __name__ == "__main__":
    # benchmark_on(
    #     dataset_name="orionweller/LIMIT-small",
    #     index_dir="./dataset/limit_small_index",
    #     top_ks=[2, 10, 20],
    #     is_first_run=True,
    # )

    # Results:
    # Recall@2 = 0.9975
    # Recall@10 = 1.0
    # Recall@20 = 1.0

    dataset_types = [
        "dense",
        "cyclic",
        "random",
        "disjoint",
    ]
    for dataset_type in dataset_types:
        print(f"Dataset type: {dataset_type}")
        benchmark_on(
            dataset_dir=f"./dataset/limit_small_{dataset_type}",
            index_dir=f"./dataset/limit_small_{dataset_type}",
            top_ks=[2, 10, 100],
            is_first_run=True,
        )
        print("-----------------------------------------------------------")

    # Dataset type: dense
    # Recall@2 = 1.0
    # Recall@10 = 1.0
    # Recall@20 = 1.0
    # Recall@100 = 1.0
    # -----------------------------------------------------------
    # Dataset type: cyclic
    # Recall@2 = 0.9782608695652174
    # Recall@10 = 1.0
    # Recall@20 = 1.0
    # Recall@100 = 1.0
    # -----------------------------------------------------------
    # Dataset type: random
    # Recall@2 = 1.0
    # Recall@10 = 1.0
    # Recall@20 = 1.0
    # Recall@100 = 1.0
    # -----------------------------------------------------------
    # Dataset type: disjoint
    # Recall@2 = 0.9782608695652174
    # Recall@10 = 1.0
    # Recall@20 = 1.0
    # Recall@100 = 1.0
    # -----------------------------------------------------------
