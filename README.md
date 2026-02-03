# embedding_dimension_limit

This repository contains my reproduction of the paper [_On the Theoretical Limitations of Embedding-Based Retrieval_](https://arxiv.org/abs/2508.21038v1), completed as the final Information Retrieval assignment for the first year of the **Data Analysis and Modelling** master’s program.

## Structure

**Notebooks**: Located in `./notebooks`. Each notebook has its own dependencies, installable via:

```sh
pip install -r ./notebooks/<notebook_name>.requirements.txt
```

**Python Scripts**:

- `bm25_benchmark.py`: BM25 benchmark; requires a [Milvus Lite](https://milvus.io/docs/milvus_lite.md) local instance.
- `colbert_benchmark.py`: [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) benchmark; requires `pylate` for index creation.

- Install script dependencies via:

  ```sh
  uv sync
  ```

## Experimental Setup

Experiments were conducted on:

- NVIDIA T4 and P100 GPUs (Kaggle)
- NVIDIA GeForce GTX 1650 (local machine)

## Usage

The accompanying presentation explains the results, the theory behind the paper, and the reproduced experiments. Additional documentation and examples will be added here soon.
