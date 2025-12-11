import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== 1️⃣ Khởi tạo retriever ======
def get_retriever(persist_dir="chroma_db", embedding_model="sentence-transformers/all-MiniLM-L6-v2", k=3):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": k})

# ====== 2️⃣ Hàm gọi GPT-3.5 để sinh câu trả lời ======
def generate_answer(question, context):
    prompt = f"""
You are an expert assistant representing Arbin Instruments.
Use ONLY the following context from Arbin's website to answer the question accurately.

Context:
{context}

Question: {question}

If the answer is not found in the context, reply:
"The information is not available on Arbin's website."
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ====== 3️⃣ Chạy test tự động ======
def run_autotest(input_file="data/arbin_eval_extended.json", output_file="data/arbin_eval_results.json"):
    retriever = get_retriever()
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        print(f"\n🧠 [{i}/{len(dataset)}] Processing: {q}")
        docs = retriever.get_relevant_documents(q)
        retrieved_docs = [d.page_content for d in docs]
        context = "\n".join(retrieved_docs)

        answer = generate_answer(q, context)
        print(f"→ Chatbot answer: {answer[:120]}...")

        item["retrieved_docs"] = retrieved_docs
        item["chatbot_answer"] = answer
        results.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ AutoTest Completed! Results saved to: {output_file}")

# ====== Run ======
if __name__ == "__main__":
    run_autotest()
