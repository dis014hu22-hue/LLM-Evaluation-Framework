import pandas as pd
import pickle
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

with open("vectordb/chunks.pkl", "rb") as f:
    chunks=pickle.load(f)
embedding_model = HuggingFaceEmbeddings (model_name = "BAAI/bge-small-en-v1.5")

vector_db = FAISS.load_local("vectordb",embedding_model,allow_dangerous_deserialization=True)

prompt = PromptTemplate(input_variables=["query", "context"], 
    template = """
You are an Enterprise Knowledge Assistant.

Rules:
1. Answer ONLY from the provided context.
2. Do not use outside knowledge.
3. If the answer does not exist in the context, say:
   "Not found in documents."
4. Mention only information supported by the context.
5. Keep the answer under 80 words.

Question:
{query}

Context:
{context}

Answer:
""")
llm = ChatOllama(model="qwen2.5:7B")
parser = StrOutputParser()
rag_chain = prompt|llm|parser

def dense_retrieve(query):
    return vector_db.similarity_search(
        query,
        k=3
    )

def ask_question(query):
    top_docs = dense_retrieve(query)
    context = ""
    retrieved_chunks = []
    sources = set()

    for doc in top_docs:

        context += doc.page_content + "\n\n"

        retrieved_chunks.append(doc.page_content)

        sources.add(
            f"{doc.metadata['source']}|page{doc.metadata['page']+1}"
        )

    final_result = rag_chain.invoke(
        {
            "query": query,
            "context": context
        }
    )

    return (
        final_result,
        list(retrieved_chunks),
        "\n".join(sorted(sources))
    )

df = pd.read_csv("dataset/evaluation_dataset.csv")
results = []

for _, row in df.iterrows():
    question = row["question"]
    answer, context_chunks, sources = ask_question(question)
    results.append(
        {
            "id": row["id"],
            "question": question,
            "expected_answer": row["expected_answer"],
            "actual_answer": answer,
            "retrieved_context": str(context_chunks),
            "sources": sources
        }
    )
    print(f"Completed {row['id']}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/dense_results.csv",
    index=False)

print("Results saved successfully")