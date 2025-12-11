# 🤖 AZVISION Chatbot

Hệ thống **AI Chatbot tích hợp RAG (Retrieval-Augmented Generation)** được phát triển bởi **AZVISION**, hỗ trợ hỏi đáp thông minh dựa trên dữ liệu nội bộ.  
Ứng dụng này kết hợp giữa **FastAPI (backend)**, **React (frontend)**, và **ChromaDB (vector store)** để cung cấp trải nghiệm hội thoại mượt mà và chính xác.

---

## 🧩 Cấu trúc thư mục
```bash
azvision-chatbot/
├── chroma_db/ # Cơ sở dữ liệu vector lưu embedding (Chroma)
├── documents/ # Thư mục chứa tài liệu gốc (PDF, DOCX, JSON, v.v.)
│
├── data_layer/ # Lớp quản lý dữ liệu & xử lý trước
│ ├── init.py
│ ├── web_crawler.py # Trình thu thập dữ liệu từ web (crawl nội dung)
│ ├── document_loader.py # Nạp tài liệu từ thư mục documents/
│ ├── preprocessor.py # Tiền xử lý, làm sạch và chia nhỏ văn bản
│ ├── data_manager.py # Quản lý pipeline dữ liệu (load, clean, chunk, save)
│ └── vector_store.py # Tạo và truy vấn VectorStore (ChromaDB...)
│
├── ai_core/ # Lõi xử lý AI
│ ├── init.py
│ ├── llm_chain.py # Chuỗi LLM (tích hợp mô hình GPT / OpenAI / Local)
│ ├── retrieval_qa.py # Kết hợp RAG: truy vấn ngữ cảnh + sinh câu trả lời
│ ├── memory_manager.py # Quản lý bộ nhớ hội thoại (conversation memory), Lưu lịch sử chat, tóm tắt ngữ cảnh và hỗ trợ multi-turn conversation
│ ├── parsers.py # Phân tích và chuẩn hóa kết quả LLM
│ ├── prompts.py # Template cho prompt RAG
│ └── nlu_processor.py # Phân tích ngôn ngữ tự nhiên (intent, entity)
│
├── api/ # Backend API (FastAPI)
│ ├── init.py
│ ├── main.py # Điểm khởi động API
│ └── schemas.py # Định nghĩa schema (Request / Response)
│
├── frontend/ # Giao diện người dùng (ReactJS)
│ ├── .env # Cấu hình biến môi trường frontend
│ ├── public/
│ │ └── index.html
│ └── src/
│ ├── components/ # Các thành phần React
│ │ ├── Chat.jsx # Thành phần chính hiển thị hội thoại
│ │ ├── Header.jsx # Thanh tiêu đề chatbot
│ │ └── InputArea.jsx # Ô nhập liệu và nút gửi tin nhắn
│ ├── App.jsx # Ứng dụng React chính
│ ├── index.js # Điểm vào frontend
│ └── styles.css # Giao diện & CSS
│
├── .env # Biến môi trường backend (API keys, DB path)
├── requirements.txt # Thư viện cần thiết
└── README.md # Hướng dẫn này

# Giải thích luồng xử lý:

Frontend
→ Người dùng nhập câu hỏi tại InputArea.jsx → gửi đến API /chat.
→ Kết quả hiển thị trong Chat.jsx.

API (FastAPI)
→ Nhận yêu cầu, tạo ChatRequest object (schemas.py).
→ Gọi retrieval_qa.generate_answer() trong AI Core.

AI Core

nlu_processor.py: phân tích intent, entity.

memory_manager.py: lấy ngữ cảnh hội thoại trước đó.

retrieval_qa.py: tìm context liên quan trong Vector Store.

llm_chain.py: tạo câu trả lời từ LLM dựa trên context + prompt.

parsers.py: định dạng lại kết quả đầu ra.

Data Layer

Xây dựng vector database từ tài liệu gốc (documents/) qua pipeline:
web_crawler → loader → preprocessor → vector_store → chroma_db/.

Vector DB (Chroma)

Lưu trữ toàn bộ embedding và cung cấp API truy vấn tương tự (semantic search).


```bash

1: Cài đặt dependencies
pip install -r requirements.txt
2: Cấu hình .env
Tạo file .env trong thư mục gốc:

OPENAI_API_KEY=your_api_key_here
CHROMA_DB_PATH=./chroma_db
DOCS_PATH=./documents
MODEL_NAME=gpt-3.5-turbo

🚀 Chạy hệ thống
🧠 1. Xử lý dữ liệu đầu vào (preprocessing)
Chạy pipeline làm sạch và chunk dữ liệu:

python -m data_layer.runs.run_1_ingest_documents         # Crawl dữ liệu
python -m data_layer.runs.run_2_web_ingestion     # Load file
python -m data_layer.runs.run_3_preprocess_chunks        # Clean + Chunk
python -m data_layer.runs.run_4_embed_store        # Build vector DB
⚡ 2. Khởi chạy backend (FastAPI)

uvicorn api.main:app --reload
Mặc định API sẽ chạy ở http://127.0.0.1:8000.

💬 3. Chạy frontend (React)

cd frontend
npm install
npm start
Giao diện sẽ chạy ở http://localhost:3000.

🔍 4️⃣ Kiểm thử Chatbot
Gửi câu hỏi về nội dung trong thư mục documents/

Chatbot sẽ:

Truy vấn vector store để tìm đoạn liên quan nhất

Gửi ngữ cảnh + câu hỏi vào LLM

Trả về câu trả lời tự nhiên, chính xác và có nguồn trích dẫn (nếu bật)

🧠 Workflow RAG
text
Copy code
[User Question]
      │
      ▼
[NLU Processor → Intent + Entity]
      │
      ▼
[Retriever → Query ChromaDB]
      │
      ▼
[LLM Chain → Combine Context + Prompt]
      │
      ▼
[LLM Response Parser → Clean Output]
      │
      ▼
[Frontend Chat Interface → Display Answer]
🧪 Đánh giá độ chính xác (Evaluation)

step#1: Sinh tự động câu trả lời GPT-3.5 cho toàn bộ test
python ai_core/rag_autotest.py 
step#2: Tính độ chính xác của chatbot
python evaluate_rag_gpt35.py

Mỗi câu hỏi trong test file gồm:

json
Copy code
{
  "question": "What battery testing systems does Arbin offer?",
  "expected_answer": "Arbin offers multi-channel battery test systems such as..."
}


workflow: 
[1] Thu thập dữ liệu → web_crawler.py
      ↓
[2] Nạp & Tiền xử lý → document_loader.py + preprocessor.py
      ↓
[3] Xây dựng Vector Store → vector_store.py (ChromaDB)
      ↓
[4] API FastAPI nhận câu hỏi người dùng
      ↓
[5] memory_manager lấy lịch sử hội thoại
      ↓
[6] retrieval_qa truy vấn vector store → tìm ngữ cảnh
      ↓
[7] llm_chain + prompts + parsers → sinh câu trả lời
      ↓
[8] memory_manager lưu hội thoại
      ↓
[9] frontend hiển thị kết quả