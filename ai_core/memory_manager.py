# ============================================================
# ARBIN MEMORY MANAGER
# Mục tiêu: Quản lý bộ nhớ hội thoại (conversation memory) cho chatbot Arbin Instruments
# Tính năng nổi bật:
# - Lưu và khôi phục lịch sử hội thoại (conversation context)
# - Theo dõi intent, entity, và ngữ cảnh sản phẩm/thuật ngữ kỹ thuật
# - Tối ưu cho các session đa người dùng (session-based memory)
# ============================================================

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class ArbinMemoryManager:
    """
    Bộ quản lý bộ nhớ (Memory Manager) tối ưu cho chatbot Arbin Instruments
    
    Bao gồm:
    - Conversation memory với giới hạn context window (chỉ giữ k tin nhắn gần nhất)
    - Intent tracking: theo dõi intent gần nhất của người dùng
    - Product context tracking: lưu sản phẩm được đề cập trong hội thoại
    - Technical context: lưu các thuật ngữ và vấn đề kỹ thuật đã nói đến
    - Session management: theo dõi thời gian, số lượng truy vấn, metadata...
    """

    def __init__(self, k=5):
        """
        Args:
            k: Số lượng tin nhắn được lưu trong bộ nhớ hội thoại (window size)
        """
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}  # Lưu memory riêng cho từng session
        self.session_data: Dict[str, Dict[str, Any]] = {}              # Lưu dữ liệu session mở rộng (intent, context...)
        self.k = k  # Giới hạn số lượng tin nhắn được lưu
        
        # Khởi tạo cấu trúc mẫu cho session_data
        self._init_session_template()

    def _init_session_template(self):
        """Khởi tạo template mặc định cho session_data"""
        self.session_template = {
            "created_at": datetime.now().isoformat(),
            "last_intent": None,                     # Intent cuối cùng người dùng hỏi
            "last_question": "",                     # Câu hỏi gần nhất
            "last_product_mentioned": None,          # Tên sản phẩm gần nhất được nhắc đến
            "technical_context": {                   # Ngữ cảnh kỹ thuật đã thảo luận
                "products_discussed": [],
                "technical_terms": [],
                "specifications_requested": [],
                "issues_discussed": []
            },
            "conversation_flow": [],                 # Lịch sử flow hội thoại (intent, entity, timestamp)
            "metadata": {                            # Metadata của session
                "language": "en",
                "user_type": "unknown",
                "query_count": 0
            }
        }

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """
        Lấy memory cho một session cụ thể (hoặc tạo mới nếu chưa có)
        """
        if session_id not in self.memories:
            # Tạo mới memory dạng cửa sổ (chỉ giữ k tin nhắn gần nhất)
            memory = ConversationBufferWindowMemory(
                k=self.k,
                return_messages=True,
                memory_key="chat_history",
                output_key="output",
                input_key="input"
            )
            self.memories[session_id] = memory
            
            # Khởi tạo dữ liệu session tương ứng
            if session_id not in self.session_data:
                self.session_data[session_id] = self.session_template.copy()
                self.session_data[session_id]["created_at"] = datetime.now().isoformat()
        
        return self.memories[session_id]

    def save_context(self, session_id: str, user_input: str, 
                    assistant_output: str, intent: str = None,
                    entities: Dict[str, Any] = None):
        """
        Lưu ngữ cảnh hội thoại và metadata cho mỗi session.
        
        Args:
            session_id: ID của session hiện tại
            user_input: Câu hỏi người dùng
            assistant_output: Câu trả lời của chatbot
            intent: Intent phát hiện được
            entities: Các entity trích xuất từ câu hỏi
        """
        try:
            memory = self.get_memory(session_id)
            
            # 1️⃣ Lưu hội thoại (input/output) vào memory LangChain
            memory.save_context(
                {"input": user_input},
                {"output": assistant_output}
            )
            
            # 2️⃣ Cập nhật session_data chi tiết
            if session_id in self.session_data:
                session = self.session_data[session_id]
                
                # Cập nhật intent cuối
                if intent:
                    session["last_intent"] = intent
                
                # Cập nhật câu hỏi cuối
                session["last_question"] = user_input
                
                # Cập nhật sản phẩm nếu có trong entities
                if entities and "product_names" in entities and entities["product_names"]:
                    product = entities["product_names"][0]
                    if product not in session["technical_context"]["products_discussed"]:
                        session["technical_context"]["products_discussed"].append(product)
                    session["last_product_mentioned"] = product
                
                # Cập nhật các thuật ngữ kỹ thuật
                if entities and "technical_terms" in entities:
                    for term in entities["technical_terms"]:
                        if term not in session["technical_context"]["technical_terms"]:
                            session["technical_context"]["technical_terms"].append(term)
                
                # Cập nhật các thông số kỹ thuật được yêu cầu
                if entities and "specifications" in entities:
                    for spec in entities["specifications"]:
                        if spec not in session["technical_context"]["specifications_requested"]:
                            session["technical_context"]["specifications_requested"].append(spec)
                
                # Cập nhật các vấn đề kỹ thuật được nói đến
                if entities and "issues" in entities:
                    for issue in entities["issues"]:
                        if issue not in session["technical_context"]["issues_discussed"]:
                            session["technical_context"]["issues_discussed"].append(issue)
                
                # Lưu lại log hội thoại vào conversation_flow
                conversation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input,
                    "intent": intent,
                    "entities": entities or {}
                }
                session["conversation_flow"].append(conversation_entry)
                
                # Cập nhật tổng số lượt hỏi
                session["metadata"]["query_count"] = len(session["conversation_flow"])
                
                # Giới hạn conversation_flow để tránh phình to (chỉ giữ 50 lượt gần nhất)
                if len(session["conversation_flow"]) > 50:
                    session["conversation_flow"] = session["conversation_flow"][-50:]
            
            # Log kiểm tra
            print(f"Arbin Memory: Saved context for session '{session_id}'")
            print(f"Intent: {intent}")
            print(f"Product mentioned: {entities.get('product_names', [])[:1] if entities else 'None'}")
            
        except Exception as e:
            print(f"Error saving context: {e}")

    def get_chat_history(self, session_id: str, format_type: str = "text") -> Any:
        """
        Lấy lại lịch sử hội thoại (chat history)
        
        Args:
            format_type: 
                - "text": trả về chuỗi hội thoại đơn giản
                - "structured": trả về danh sách dict (để frontend hiển thị)
        """
        try:
            memory = self.get_memory(session_id)
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            if not chat_history:
                return "" if format_type == "text" else []
            
            # Dạng văn bản
            if format_type == "text":
                history_text = ""
                for msg in chat_history:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        role = "User" if msg.type == "human" else "Assistant"
                        history_text += f"{role}: {msg.content}\n"
                return history_text
            
            # Dạng structured (cho frontend)
            else:
                structured_history = []
                for msg in chat_history:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        try:
                            safe_content = str(msg.content)
                        except Exception:
                            safe_content = json.dumps(msg.content, ensure_ascii=False)
                        
                        entry = {
                            "sender": "user" if msg.type == "human" else "bot",
                            "text": safe_content,
                            "type": msg.type
                        }
                        structured_history.append(entry)
                return structured_history

        except Exception as e:
            print(f"Error getting chat history: {e}")
            return "" if format_type == "text" else []

    # ======== Các hàm getter nhanh cho session data ========
    def get_last_intent(self, session_id: str) -> str:
        """Lấy intent cuối cùng của người dùng"""
        if session_id in self.session_data:
            return self.session_data[session_id].get("last_intent", "")
        return ""
    
    def get_last_question(self, session_id: str) -> str:
        """Lấy câu hỏi cuối cùng của user"""
        if session_id in self.session_data:
            return self.session_data[session_id].get("last_question", "")
        return ""
    
    def get_last_product_mentioned(self, session_id: str) -> Optional[str]:
        """Lấy sản phẩm cuối cùng được nhắc đến"""
        if session_id in self.session_data:
            return self.session_data[session_id].get("last_product_mentioned")
        return None
    
    def get_technical_context(self, session_id: str) -> Dict[str, Any]:
        """Lấy toàn bộ technical context (các thuật ngữ/sản phẩm/vấn đề kỹ thuật đã nói)"""
        if session_id in self.session_data:
            return self.session_data[session_id].get("technical_context", {})
        return {}
    
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Trả về danh sách đối tượng message gốc của LangChain"""
        try:
            memory = self.get_memory(session_id)
            return memory.chat_memory.messages
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []

    # ======== Xóa bộ nhớ =========
    def clear_memory(self, session_id: str):
        """Xóa bộ nhớ cho một session cụ thể"""
        if session_id in self.memories:
            del self.memories[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        print(f"Cleared memory for session '{session_id}'")
    
    def clear_all_memories(self):
        """Xóa tất cả bộ nhớ của toàn hệ thống"""
        self.memories.clear()
        self.session_data.clear()
        print("Cleared all memories")

    # ======== Truy xuất dữ liệu mở rộng ========
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Lấy thông tin chi tiết của session (bao gồm thống kê bộ nhớ)"""
        if session_id in self.session_data:
            session = self.session_data[session_id].copy()
            
            # Lấy thêm thông tin thống kê memory
            memory = self.get_memory(session_id)
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", [])
            
            session["memory_stats"] = {
                "message_count": len(chat_history),
                "has_memory": len(chat_history) > 0,
                "window_size": self.k
            }
            
            return session
        
        return {"error": "Session not found"}
    
    def update_metadata(self, session_id: str, key: str, value: Any):
        """Cập nhật metadata (ví dụ: ngôn ngữ, loại user,...)"""
        if session_id in self.session_data:
            self.session_data[session_id]["metadata"][key] = value

    def get_conversation_summary(self, session_id: str) -> str:
        """Tạo tóm tắt (summary) nội dung hội thoại"""
        if session_id not in self.session_data:
            return "No session data available"
        
        session = self.session_data[session_id]
        conversation_flow = session.get("conversation_flow", [])
        
        if not conversation_flow:
            return "No conversation yet"
        
        summary_parts = []
        summary_parts.append(f"Conversation Summary for session '{session_id}':")
        summary_parts.append(f"- Started: {session['created_at']}")
        summary_parts.append(f"- Total queries: {session['metadata']['query_count']}")
        
        # Tóm tắt sản phẩm, intent, chủ đề gần nhất
        products = session['technical_context']['products_discussed']
        if products:
            summary_parts.append(f"- Products discussed: {', '.join(products)}")
        
        intents = set(entry['intent'] for entry in conversation_flow if entry.get('intent'))
        if intents:
            summary_parts.append(f"- Main intents: {', '.join(intents)}")
        
        if len(conversation_flow) >= 2:
            recent = conversation_flow[-2:]
            summary_parts.append("- Recent topics:")
            for entry in recent:
                summary_parts.append(f"  • {entry['user_input'][:100]}...")
        
        return "\n".join(summary_parts)
    
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Xuất toàn bộ dữ liệu session (gửi ra frontend hoặc lưu file)"""
        if session_id not in self.session_data:
            return {"error": "Session not found"}
        
        session = self.session_data[session_id].copy()
        
        # Thêm lịch sử chat dạng structured (đã JSON-safe)
        chat_history = self.get_chat_history(session_id, "structured")
        session["chat_history"] = [
            {
                "sender": str(msg.get("sender", "bot")),
                "text": str(msg.get("text", "")),
                "type": str(msg.get("type", "assistant"))
            }
            for msg in chat_history
        ]
        
        return session
    
    def import_session_data(self, session_id: str, data: Dict[str, Any]):
        """Import dữ liệu session (phục hồi hội thoại cũ từ file hoặc DB)"""
        self.session_data[session_id] = data
        
        # Nếu có lịch sử hội thoại, khởi tạo lại memory tương ứng
        if "chat_history" in data:
            self._initialize_memory_from_history(session_id, data["chat_history"])
        
        print(f"Imported session data for '{session_id}'")
    
    def _initialize_memory_from_history(self, session_id: str, chat_history: List[Dict]):
        """Khởi tạo lại memory LangChain từ chat_history (đã lưu trước đó)"""
        memory = self.get_memory(session_id)
        
        # (Hiện tại placeholder — có thể mở rộng trong tương lai)
        for entry in chat_history:
            if entry["role"] == "user":
                pass
            elif entry["role"] == "assistant":
                pass
        
        print(f"Initialized memory from history for '{session_id}'")


# ============================================================
# Factory function để khởi tạo Memory Manager
# ============================================================
def create_arbin_memory_manager(k: int = 5) -> ArbinMemoryManager:
    """Factory function để tạo đối tượng ArbinMemoryManager"""
    return ArbinMemoryManager(k=k)
