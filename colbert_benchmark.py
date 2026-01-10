import os

os.environ["TQDM_DISABLE"] = "1"

from pylate import indexes, models, retrieve
from datasets import load_dataset

dataset_name = "orionweller/LIMIT-small"
qrels = load_dataset(dataset_name, "default", split="all").to_pandas()
corpus = load_dataset(dataset_name, "corpus", split="all").to_pandas()
queries = load_dataset(dataset_name, "queries", split="all").to_pandas()


model = models.ColBERT(
    model_name_or_path="lightonai/GTE-ModernColBERT-v1",
    device="cuda",
)


def store_docs(
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
    show_progress: bool = False,
    index_folder: str = "./dataset/limit_small_index",
    index_name: str = "limit_small_index",
):
    index = indexes.Voyager(
        index_folder=index_folder,
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


def benchmark(recall_at: int) -> float:
    retrieved_docs = retrieve_docs(top_k=recall_at)
    relevant_docs = qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()

    total = 0
    for qid in queries["_id"]:
        total += len(retrieved_docs[qid] & relevant_docs[qid]) / len(relevant_docs[qid])

    return total / len(relevant_docs)


if __name__ == "__main__":
    store_docs()
    for n_docs in [2, 10, 20]:
        print(f"Recall@{n_docs} = {benchmark(recall_at=n_docs)}")

    # Results:
    # Recall@2 = 0.9975
    # Recall@10 = 1.0
    # Recall@20 = 1.0
