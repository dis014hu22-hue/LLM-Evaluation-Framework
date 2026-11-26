from deepeval.models import OllamaModel
import pandas as pd
import ast
import os
os.environ[    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (    AnswerRelevancyMetric,    FaithfulnessMetric)
import time

judge = OllamaModel( model="qwen2.5:7b")
selected_ids = ["Q01","Q05","Q10","Q15","Q20","Q24","Q31","Q39","Q45","Q49"]

answer_relevancy = AnswerRelevancyMetric(model=judge)
faithfulness = FaithfulnessMetric(model=judge)

#-----------------------------------------------------------------------------------------------#
#                           Rerank evaluation 
#-----------------------------------------------------------------------------------------------#
start_time = time.time()
df = pd.read_csv("dataset/eval_subset.csv")
results = []
a=0
for idx, row in df.iterrows():

    context = ast.literal_eval(row["retrieved_context"])
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["actual_answer"],
        expected_output=row["expected_answer"],
        retrieval_context=context,
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)    

    results.append({
        "id": row["id"],
        "answer_relevancy": answer_relevancy.score,
        "faithfulness": faithfulness.score
        
    })
    a+= 1
    print (a, " ")
    print(f"Completed {row['id']}")
    pd.DataFrame(results).to_csv(
    "results/rerank_metrics.csv",
    index=False )

print("Rerank Evaluation Complete")

end_time = time.time()
total_minutes = (end_time - start_time) / 60
print(f"\nTotal Runtime: {total_minutes:.2f} minutes")

#-----------------------------------------------------------------------------------------------#
#                           Dense evaluation 
#-----------------------------------------------------------------------------------------------#
print("\n" + "="*60)
print("STARTING DENSE EVALUATION")
print("="*60)

start_time = time.time()
df = pd.read_csv("results/dense_results.csv")
df = df[df["id"].isin(selected_ids)]
results = []
a=0
for idx, row in df.iterrows():

    context = ast.literal_eval(row["retrieved_context"])
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["actual_answer"],
        expected_output=row["expected_answer"],
        retrieval_context=context,
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)    

    results.append({
        "id": row["id"],
        "answer_relevancy": answer_relevancy.score,
        "faithfulness": faithfulness.score
        
    })
    a+= 1
    print (a, " ")
    print(f"Completed {row['id']}")
    pd.DataFrame(results).to_csv(
    "results/dense_metrics.csv",
    index=False )

print("Dense Evaluation Complete")

end_time = time.time()
total_minutes = (end_time - start_time) / 60
print(f"\nTotal Runtime: {total_minutes:.2f} minutes")

import winsound

for _ in range(3):
    winsound.Beep(1200, 500)

#-----------------------------------------------------------------------------------------------#
#                           Hybrid evaluation 
#-----------------------------------------------------------------------------------------------#
print("\n" + "="*60)
print("STARTING HYBRID EVALUATION")
print("="*60)

start_time = time.time()
df = pd.read_csv("results/hybrid_results.csv")
df = df[df["id"].isin(selected_ids)]

results = []
a=0
for idx, row in df.iterrows():

    context = ast.literal_eval(row["retrieved_context"])
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["actual_answer"],
        expected_output=row["expected_answer"],
        retrieval_context=context,
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)    

    results.append({
        "id": row["id"],
        "answer_relevancy": answer_relevancy.score,
        "faithfulness": faithfulness.score
        
    })
    a+= 1
    print (a, " ")
    print(f"Completed {row['id']}")
    pd.DataFrame(results).to_csv(
    "results/hybrid_metrics.csv",
    index=False )

print("Hybrid Evaluation Complete")

end_time = time.time()
total_minutes = (end_time - start_time) / 60
print(f"\nTotal Runtime: {total_minutes:.2f} minutes")

for _ in range(5):
    winsound.Beep(1800, 500)