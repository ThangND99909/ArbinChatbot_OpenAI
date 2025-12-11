from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from typing import List
import math

from .schemas import ChatRequest, ChatResponse, DocumentUploadResponse, HealthResponse
from ai_core.retrieval_qa import ArbinRetrievalQA
from ai_core.nlu_processor import NLUProcessor
#from ai_core.llm_chain import LLMManager
from ai_core.llm_chain import get_llm_manager
from data_layer.vector_store import VectorStore
from data_layer.web_crawler import WebCrawler
from data_layer.document_loader import DocumentProcessor  # SỬA: document_processor thay vì document_loader
from data_layer.preprocessor import TextPreprocessor

app = FastAPI(title="Arbin Chatbot API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
#llm_manager = LLMManager()
llm_manager = get_llm_manager(use_openai=True)
qa_system = ArbinRetrievalQA(llm_manager.llm, vector_store)
nlu_processor = NLUProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    print("Arbin Chatbot API starting up...")
    
    # Check if vector store is empty, load initial data if needed
    try:
        stats = vector_store.get_collection_stats()  # SỬA: dùng get_collection_stats
        count = stats.get('total_documents', 0)
        print(f"Vector store contains {count} documents")
    except Exception as e:
        print(f"Error checking vector store: {e}")

@app.get("/", response_model=dict)
async def root():
    return {"message": "Arbin Chatbot API", "status": "running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        stats = vector_store.get_collection_stats()  # SỬA
        count = stats.get('total_documents', 0)
        return HealthResponse(
            status="healthy",
            vector_store_count=count
        )
    except Exception as e:
        return HealthResponse(
            status=f"error: {str(e)}",
            vector_store_count=0
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages from frontend"""
    try:
        # === 1️⃣ Gọi pipeline RAG ===
        result = qa_system.get_response(
            question=request.message,
            session_id=request.session_id or "default",
            language=None  # Auto-detect ngôn ngữ
        )

        # === 2️⃣ Chuẩn hóa câu trả lời ===
        answer = result.get("answer", "")
        if not isinstance(answer, str):
            if isinstance(answer, dict) and "text" in answer:
                answer = str(answer["text"])
            else:
                answer = str(answer)

        # === 3️⃣ Lọc và xử lý sources ===
        raw_sources = result.get("sources", [])
        safe_sources = []

        for s in raw_sources:
            if not isinstance(s, dict):
                continue

            title = str(s.get("title", "")).strip()
            url = str(s.get("url", s.get("source", ""))).strip()
            score_raw = s.get("relevance_score", s.get("score", 0))

            # 🧠 Làm sạch toàn bộ các giá trị lỗi
            try:
                score = float(score_raw)
                if math.isnan(score) or math.isinf(score) or score < 0:
                    score = 0.0
                elif score > 1.0:
                    score = 1.0
            except (TypeError, ValueError):
                score = 0.0

            # 🔒 Nếu không có tên hoặc score = 0 thì bỏ qua
            if not title or score <= 0:
                continue

            # 🔗 Nếu không có link, thử nối link nội bộ
            if not url.startswith("http") and title.endswith(".pdf"):
                url = f"/static/docs/{title}"

            # ✅ Format % và icon tin cậy
            score_percent = f"{int(score * 100)}%"
            if score >= 0.8:
                icon = "✅"
            elif score >= 0.6:
                icon = "🟡"
            else:
                icon = "⚠️"

            safe_sources.append({
                "title": title,
                "url": url,
                "score": f"{score_percent} {icon}"
            })

        # === 4️⃣ Nếu không có nguồn hợp lệ, bỏ luôn trường sources ===
        if not safe_sources:
            safe_sources = None

        # === 5️⃣ Chuẩn hóa response trả về frontend ===
        response_data = ChatResponse(
            answer=answer.strip(),
            sources=safe_sources,
            session_id=request.session_id or "default",
            intent=str(result.get("intent", "unknown"))
        )

        return response_data

    except Exception as e:
        print(f"❌ Lỗi trong /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ingest/website")
async def ingest_website():
    """Crawl and ingest website content"""
    try:
        crawler = WebCrawler()
        links = crawler.get_sitemap_links()
        
        documents = []
        for link in links[:5]:  # Limit to 5 pages for demo
            doc = crawler.crawl_page(link)
            if doc:
                documents.append(doc)
        
        # Preprocess and add to vector store
        preprocessor = TextPreprocessor()
        chunks = preprocessor.preprocess_documents(documents)
        vector_store.add_documents(chunks)
        
        return {"message": f"Ingested {len(documents)} web pages", "count": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/documents")
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        documents_dir = "./documents"
        os.makedirs(documents_dir, exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = os.path.join(documents_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file_path)
        
        # Process documents
        processor = DocumentProcessor(documents_dir)
        docs = processor.process_all_documents()
        
        # Preprocess and add to vector store
        preprocessor = TextPreprocessor()
        chunks = preprocessor.preprocess_documents(docs)
        vector_store.add_documents(chunks)
        
        return DocumentUploadResponse(
            message=f"Processed {len(files)} files",
            processed_count=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="frontend/public"), name="static")