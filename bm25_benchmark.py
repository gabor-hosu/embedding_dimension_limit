from pymilvus import MilvusClient, DataType, Function, FunctionType
from datasets import load_dataset

corpus = load_dataset(
    "json", data_files="./dataset/custom_limit_small/corpus.jsonl", split="all"
)
corpus = corpus.to_pandas()

qrels = load_dataset(
    "json", data_files="./dataset/custom_limit_small/qrels.jsonl", split="all"
)
qrels = qrels.to_pandas()

queries = load_dataset(
    "json", data_files="./dataset/custom_limit_small/queries.jsonl", split="all"
)
queries = queries.to_pandas()

client = MilvusClient("./dataset/custom_limit_small_milvus/milvus.db")


def store_docs():
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


def retrieve_docs(top_k: int):
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


def benchmark(recall_at: int) -> float:
    retrieved_docs = retrieve_docs(top_k=recall_at)
    relevant_docs = qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()

    total = 0
    for qid in queries["_id"]:
        total += len(retrieved_docs[qid] & relevant_docs[qid]) / len(relevant_docs[qid])

    return total / len(relevant_docs)


# store_docs()
for n_docs in [2, 10, 20]:
    print(f"Recall@{n_docs} = {benchmark(recall_at=n_docs)}")
