import pandas as pd

files = {
    "Dense":"results/dense_metrics.csv",
    "Hybrid":"results/hybrid_metrics.csv",
    "Rerank":"results/rerank_metrics.csv"
}

for name,file in files.items():

    df = pd.read_csv(file)

    print(
        name,
        round(df["answer_relevancy"].mean(),3),
        round(df["faithfulness"].mean(),3)
    )