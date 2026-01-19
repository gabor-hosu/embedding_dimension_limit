from pymilvus import MilvusClient, DataType, Function, FunctionType
from datasets import load_dataset
import pandas as pd


def store_docs(corpus: pd.DataFrame, client: MilvusClient):
    schema = client.create_schema()

    schema.add_field(
        field_name="_id", datatype=DataType.VARCHAR, max_length=20, is_primary=True
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=20000,
        enable_analyzer=True,
    )
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )

    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )

    collection_name = "corpus"
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )

    client.insert(
        collection_name,
        [{"_id": row["_id"], "text": row["text"]} for _, row in corpus.iterrows()],
    )


def retrieve_docs(top_k: int, queries: pd.DataFrame, client: MilvusClient):
    search_params = {
        "params": {"drop_ratio_search": 0.2},
    }

    results = client.search(
        collection_name="corpus",
        data=queries["text"].to_list(),
        anns_field="sparse",
        output_fields=[
            "text"
        ],  # Fields to return in search results; sparse field cannot be output
        limit=top_k,
        search_params=search_params,
    )

    retrieved_docs = {
        f"query_{idx}": {hit["id"] for hit in result}
        for idx, result in enumerate(results)
    }

    return retrieved_docs


def benchmark(
    recall_at: int,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    client: MilvusClient,
) -> float:
    retrieved_docs = retrieve_docs(top_k=recall_at, queries=queries, client=client)
    relevant_docs = qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()

    total = 0
    for qid in queries["_id"]:
        total += len(retrieved_docs[qid] & relevant_docs[qid]) / len(relevant_docs[qid])

    return total / len(relevant_docs)


def benchmark_on(
    db_collection_path: str,
    top_ks: list[int],
    dataset_name: str = None,
    dataset_dir: str = None,
    is_first_run: bool = False,
):
    if not dataset_name and not db_collection_path:
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

    client = MilvusClient(db_collection_path)

    if is_first_run:
        store_docs(corpus=corpus, client=client)

    for top_k in top_ks:
        print(
            f"Recall@{top_k} = {benchmark(recall_at=top_k, queries=queries, qrels=qrels, client=client)}"
        )


if __name__ == "__main__":
    # benchmark_on(
    #     dataset_name="orionweller/LIMIT-small",
    #     db_collection_path="./dataset/limit_small_milvus/milvus.db",
    #     top_ks=[2, 10, 20],
    # )

    # Results:
    # Recall@2 = 0.9915
    # Recall@10 = 1.0
    # Recall@20 = 1.0

    dataset_types = ["dense", "cyclic", "random", "disjoint"]
    for dataset_type in dataset_types:
        print(f"Dataset type: {dataset_type}")
        benchmark_on(
            dataset_dir=f"./dataset/limit_small_{dataset_type}",
            db_collection_path=f"./dataset/limit_small_{dataset_type}/milvus.db",
            top_ks=[2],
            is_first_run=False,
        )
        print("-----------------------------------------------------------")

    # Results:
    # Dataset type: dense
    # Recall@2 = 0.9565217391304348
    # -----------------------------------------------------------
    # Dataset type: cyclic
    # Recall@2 = 0.9565217391304348
    # -----------------------------------------------------------
    # Dataset type: random
    # Recall@2 = 0.9347826086956522
    # -----------------------------------------------------------
    # Dataset type: disjoint
    # Recall@2 = 0.9130434782608695
    # -----------------------------------------------------------
