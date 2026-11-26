## Evaluation Observations

Manual inspection of low-scoring examples revealed several useful findings:

### Case 1 – Q20 (Rerank)
The model returned "Not found in documents" despite the required information being present in the retrieved context.

Impact:
- High retrieval relevance
- Lower faithfulness
- Indicates answer-generation failure rather than retrieval failure

### Case 2 – Q39 (Rerank)
Retrieved context clearly contained information about work hours and overtime, but the generated answer omitted several required details.

Impact:
- Relevant answer
- Incomplete grounding
- Reduced faithfulness score

### Case 3 – Q49 (Hybrid)
The expected response was "Not Found in Document."
The model incorrectly inferred a probation period from unrelated text mentioning a 4‑month certificate program.

Impact:
- Hallucination caused by misleading context
- Demonstrates why faithfulness evaluation is important

## Evaluation Results

Evaluation was performed using DeepEval and LLM-as-a-Judge (Qwen2.5:7B via Ollama).

Benchmark:
- 50-question evaluation dataset
- 10-question representative benchmark subset
- Metrics: Answer Relevancy, Faithfulness

| Configuration | Relevancy | Faithfulness |
|--------------|-----------|-------------|
| Dense Retrieval | 0.95 | 0.767 |
| Hybrid Retrieval | 0.95 | 0.800 |
| Hybrid + Rerank | 1.00 | 0.725 |

### Runtime

| Configuration | Runtime |
|--------------|---------|
| Dense | 24 min |
| Hybrid | 33 min |
| Rerank | 28 min |

### Key Findings

- Hybrid retrieval produced the highest faithfulness score.
- Reranking achieved perfect answer relevancy on the benchmark subset.
- Evaluation uncovered multiple grounding failures that would not be visible through manual spot checks alone.
- Not-found questions proved valuable for identifying hallucinations.