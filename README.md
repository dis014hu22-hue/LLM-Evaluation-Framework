# LLM Evaluation Framework

An evaluation framework for Retrieval-Augmented Generation (RAG) systems built using DeepEval and LLM-as-a-Judge.

The framework benchmarks multiple retrieval strategies and measures answer quality using automated evaluation metrics.

## Features

- DeepEval Integration
- LLM-as-a-Judge Evaluation
- Automated Benchmark Dataset
- Dense Retrieval Evaluation
- Hybrid Retrieval Evaluation
- Hybrid + Reranking Evaluation
- Faithfulness Measurement
- Answer Relevancy Measurement
- CSV-Based Reporting
- Local Evaluation using Qwen2.5 via Ollama

## Evaluation Dataset

- Source: Human Resource Policy Document
- Total Questions: 50
- Evaluation Subset: 10 Questions

## Retrieval Configurations

1. Dense Retrieval (FAISS)
2. Hybrid Retrieval (FAISS + BM25)
3. Hybrid + Reranking (FAISS + BM25 + Cross Encoder)

## Results
| Configuration | Relevancy | Faithfulness |
|--------------|-----------|-------------|
| Dense | 0.95 | 0.767 |
| Hybrid | 0.95 | 0.800 |
| Hybrid + Rerank | 1.00 | 0.725 |

##Architecture

Evaluation Dataset
        ↓
Dense / Hybrid / Rerank
        ↓
Generated Answers
        ↓
DeepEval
        ↓
Answer Relevancy
Faithfulness
        ↓
Comparison Report
