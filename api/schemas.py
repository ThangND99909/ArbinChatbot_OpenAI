from pydantic import BaseModel
from typing import List, Optional


# ====== 🔹 Mỗi nguồn tham khảo (source item) ======
class SourceItem(BaseModel):
    title: str
    url: Optional[str] = None          # Đường dẫn đến file hoặc trang web
    score: Optional[str] = None        # Ví dụ: "82% ✅"


# ====== 🔹 Yêu cầu chat từ frontend ======
class ChatRequest(BaseModel):
    message: str                       # Tin nhắn người dùng gửi
    session_id: Optional[str] = None   # ID phiên hội thoại (dùng cho memory)


# ====== 🔹 Phản hồi chat từ backend ======
class ChatResponse(BaseModel):
    answer: str                        # Câu trả lời cuối cùng từ chatbot
    sources: Optional[List[SourceItem]] = None  # Danh sách tài liệu tham khảo (hoặc None)
    session_id: Optional[str] = None   # Giữ nguyên session ID
    intent: Optional[str] = None       # Intent đã nhận diện được (VD: specification_request)


# ====== 🔹 Kết quả upload tài liệu ======
class DocumentUploadResponse(BaseModel):
    message: str
    processed_count: int


# ====== 🔹 Kiểm tra sức khỏe hệ thống ======
class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
