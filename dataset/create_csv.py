import re
import pandas as pd
from docx import Document

# Load document
doc = Document("dataset/HR_Policy_RAG_Evaluation_50_QA.docx")

# Extract all text
full_text = "\n".join([p.text for p in doc.paragraphs])

# ------------------
# Extract Questions
# ------------------
question_pattern = r"Q(\d{2})\.\s*(.*?)(?=\nQ\d{2}\.|\n## PART II|$)"

questions = {
    qid: qtext.strip()
    for qid, qtext in re.findall(question_pattern, full_text, re.DOTALL)
}

# ------------------
# Extract Answers
# ------------------
answer_pattern = (
    r"A(\d{2})\s+(.*?)(?=\nA\d{2}\s|\n## Suggested scoring guide|$)"
)

answers = {
    aid: re.sub(r"_Evidence:.*?_", "", atext, flags=re.DOTALL).strip()
    for aid, atext in re.findall(answer_pattern, full_text, re.DOTALL)
}

# ------------------
# Build dataset
# ------------------
rows = []

for qid in sorted(questions.keys()):
    rows.append(
        {
            "id": f"Q{qid}",
            "question": questions[qid],
            "expected_answer": answers.get(qid, "")
        }
    )

df = pd.DataFrame(rows)

output_path = "dataset/evaluation_dataset.csv"
df.to_csv(output_path, index=False)

print(f"CSV saved: {output_path}")
print(f"Rows: {len(df)}")