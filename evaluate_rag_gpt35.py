import json
import openai
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List

openai.api_key = "YOUR_OPENAI_API_KEY"  # ⚠️ nhập API key của bạn

model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ====== 1️⃣ Retrieval metrics ======

def recall_at_k(gt_docs: List[str], retrieved_docs: List[List[str]], k=3):
    hits = sum(any(gt in d for d in docs[:k]) for gt, docs in zip(gt_docs, retrieved_docs))
    return hits / len(gt_docs)

def mrr(gt_docs: List[str], retrieved_docs: List[List[str]]):
    ranks = []
    for gt, docs in zip(gt_docs, retrieved_docs):
        for rank, d in enumerate(docs, start=1):
            if gt in d:
                ranks.append(1 / rank)
                break
        else:
            ranks.append(0)
    return np.mean(ranks)

# ====== 2️⃣ Semantic accuracy (cosine similarity) ======

def semantic_accuracy(gt_answers, chatbot_answers):
    emb_gt = model_sbert.encode(gt_answers, convert_to_tensor=True)
    emb_pred = model_sbert.encode(chatbot_answers, convert_to_tensor=True)
    sims = util.cos_sim(emb_gt, emb_pred)
    return sims.diag().mean().item()

# ====== 3️⃣ Hallucination scoring using GPT-3.5 ======

def hallucination_score(question, answer, context):
    prompt = f"""
    You are evaluating a chatbot's factual accuracy.

    Question: {question}
    Context (retrieved info): {context}
    Chatbot's answer: {answer}

    Rate the factual accuracy of the chatbot's answer from 0 to 1:
    - 1 = completely correct and supported by context
    - 0.5 = partially correct, minor inaccuracies
    - 0 = incorrect or hallucinated information
    Only return the number.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return float(response.choices[0].message["content"].strip())
    except:
        return 0.0

# ====== 4️⃣ Evaluation pipeline ======

def evaluate_rag(dataset_path, k=3, weight_retrieval=0.4):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    gt_docs = [item["ground_truth"] for item in data]
    retrieved_docs = [item["retrieved_docs"] for item in data]
    gt_answers = [item["ground_truth"] for item in data]
    chatbot_answers = [item["chatbot_answer"] for item in data]

    retrieval_recall = recall_at_k(gt_docs, retrieved_docs, k)
    retrieval_mrr = mrr(gt_docs, retrieved_docs)
    generation_acc = semantic_accuracy(gt_answers, chatbot_answers)

    halluc_scores = []
    for item in data:
        context_text = " ".join(item["retrieved_docs"])
        score = hallucination_score(item["question"], item["chatbot_answer"], context_text)
        halluc_scores.append(score)
    hallucination_avg = np.mean(halluc_scores)

    final_score = (
        weight_retrieval * retrieval_recall +
        (1 - weight_retrieval) * generation_acc
    ) * hallucination_avg

    print(f"🔍 Retrieval Recall@{k}: {retrieval_recall:.3f}")
    print(f"📈 MRR: {retrieval_mrr:.3f}")
    print(f"🧠 Generation Semantic Accuracy: {generation_acc:.3f}")
    print(f"🧩 Hallucination Score (GPT-3.5): {hallucination_avg:.3f}")
    print(f"🏁 Final Weighted Accuracy: {final_score:.3f}")

if __name__ == "__main__":
    evaluate_rag("arbin_eval.json", k=3, weight_retrieval=0.4)
