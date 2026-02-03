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

## Models

Due to hardware constraints, larger models in the 7-8B parameter range were quantized to 4-bit precision. These quantized variants were created specifically for this project and are available on Hugging Face:

- [intfloat/e5-mistral-7b-instruct (4-bit quantized)](https://huggingface.co/gabor-hosu/e5-mistral-7b-instruct-bnb-4bit)
- [GritLM/GritLM-7B (4-bit quantized)](https://huggingface.co/gabor-hosu/GritLM-7B-bnb-4bit)
- [Qwen/Qwen3-Embedding-8B (4-bit quantized)](https://huggingface.co/gabor-hosu/Qwen3-Embedding-8B-bnb-4bit)

## Datasets

The original paper's qrel effect experiment datasets were not publicly available, and the full 50K-document scale exceeded available resources.  
To reproduce the experiments, smaller datasets were created by populating different qrel binary patterns with natural language, while preserving the name and attribute distributions from the original LIMIT dataset.

Available datasets:

- [LIMIT-small-dense](https://huggingface.co/datasets/gabor-hosu/LIMIT-small-dense) - Dense binary relevance: each query has multiple overlapping relevant documents.
- [LIMIT-small-disjoint](https://huggingface.co/datasets/gabor-hosu/LIMIT-small-disjoint) - Disjoint relevance: queries have non-overlapping relevant documents.
- [LIMIT-small-cyclic](https://huggingface.co/datasets/gabor-hosu/LIMIT-small-cyclic) - Cyclic relevance pattern: relevance relationships form a cycle across queries.
- [LIMIT-small-random](https://huggingface.co/datasets/gabor-hosu/LIMIT-small-random) - Random relevance assignments: queries have randomly selected relevant documents.

## Usage

The accompanying (`presentation.pdf`) presentation explains the results, the theory behind the paper, and the reproduced experiments. Additional documentation and examples will be added here soon.
