# ai_core/retrieval_qa.py - PHIÊN BẢN ARBIN
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List, Optional, Tuple
from .nlu_processor import NLUProcessor
from .memory_manager import ArbinMemoryManager
# XÓA: from .parsers import NLUOutputParser  # Không cần thiết
import traceback
import re

class ArbinRetrievalQA:
    """
    Lớp chính cho mô-đun Hỏi-Đáp (QA) trong chatbot Arbin Instruments.
    """
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        
        # Bộ nhớ hội thoại
        self.memory_manager = ArbinMemoryManager()
        
        # NLU Processor với memory integration
        self.nlu_processor = NLUProcessor(llm, memory_manager=self.memory_manager)
        
        # Thiết lập các Prompt Template
        self.setup_qa_chains()
        self._setup_language_detection()

    def _setup_language_detection(self):
        """Cấu hình language detection patterns"""
        self.VIETNAMESE_CHARS = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        self.VIETNAMESE_WORDS = [
            "của", "là", "và", "có", "được", "trong", "cho", "với", 
            "tại", "từ", "như", "về", "này", "khi", "các"
        ]
        self.VIETNAMESE_PHRASES = [
            "là gì", "bao nhiêu", "thế nào", "tại sao", 
            "ở đâu", "có thể", "làm sao"
        ]
        self.ENGLISH_PATTERNS = [
            "what", "how", "why", "when", "where", 
            "which", "can you", "could you", "please"
        ]
        self.COMMON_ENGLISH = ["hello", "hi", "hey", "greetings"]

        self.GREETING_PATTERNS = [
            # English
            "hello", "hi", "hey", "greetings",
            # Vietnamese
            "xin chào", "chào", "chào bạn", "chào bot",
            # Short forms
            "helo", "hii", "heyy"
        ]

    def detect_language(self, text: str) -> str:
        return self._detect_language(text)

    def _detect_language(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return "en"
        text_lower = text.lower().strip()

        # RULE 1: Vietnamese characters
        if any(char in text for char in self.VIETNAMESE_CHARS):
            return "vi"

        # RULE 2: Vietnamese phrases
        if any(phrase in text_lower for phrase in self.VIETNAMESE_PHRASES):
            return "vi"

        # RULE 3: Multiple Vietnamese words
        vi_words_found = [word for word in self.VIETNAMESE_WORDS if word in text_lower]
        if len(vi_words_found) >= 2:
            return "vi"

        # RULE 4: English patterns
        if any(pattern in text_lower for pattern in self.ENGLISH_PATTERNS):
            return "en"

        # RULE 5: Common English words - exact match
        words = text_lower.split()
        if any(word in self.COMMON_ENGLISH for word in words):
            return "en"

        # RULE 6: Short technical query
        if len(words) <= 3:
            technical_terms = ["bt-", "mits", "arbin", "voltage", "current", "battery"]
            if any(term in text_lower for term in technical_terms):
                return "en"

        # Default
        return "en"

    def _is_greeting(self, text: str) -> bool:
        """Cải tiến: kiểm tra greeting không dùng substring trong từ"""
        if not text or not isinstance(text, str):
            return False
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', '', text_lower)

        # Exact match
        if text_clean in self.GREETING_PATTERNS:
            return True

        # Optional: check first word exact match only
        first_word = text_clean.split()[0] if text_clean.split() else ""
        if first_word in self.GREETING_PATTERNS:
            return True

        return False
    
    
    def _resolve_language(self, question: str, user_language: str = None) -> str:
        """
        Xác định ngôn ngữ cuối cùng để dùng
        Priority: user_provided > auto_detected > default
        """
        # Priority 1: User explicitly provided
        if user_language and user_language in ["vi", "en"]:
            print(f"✓ Using user-provided language: {user_language}")
            return user_language
        
        # Priority 2: Auto-detect
        detected = self._detect_language(question)
        print(f"✓ Auto-detected language: {detected} for: '{question[:50]}...'")
        
        return detected
    
    

    # ========================== TẠO PROMPT CHO TỪNG INTENT ==========================
    def setup_qa_chains(self):
        """Thiết lập các QA chain dùng prompt từ prompts.py"""

        from ai_core.prompts import (
            qa_prompt,
            tech_support_prompt,
            comparison_prompt,
            general_support_prompt
        )

        # Mapping intent → prompt template
        self.prompt_mapping = {
            "product_inquiry": qa_prompt,
            "technical_support": tech_support_prompt,
            "specification_request": qa_prompt,
            "comparison_request": comparison_prompt,
            "application_info": qa_prompt,
            "pricing_inquiry": general_support_prompt,
            "general_info": general_support_prompt,
            "troubleshooting": tech_support_prompt,
            "other": general_support_prompt
        }

        print("✅ QA prompt chains loaded from ai_core/prompts.py")


    def _generate_response(self, question: str, context: str, intent: str,
                       language: str, chat_history: str, entities: Dict) -> str:
        """Generate response (Gemini-compatible, dùng prompts.py)"""

        # 1️⃣ Chọn ChatPromptTemplate từ mapping
        selected_prompt = self.prompt_mapping.get(intent, self.prompt_mapping["other"])

        try:
            # 2️⃣ Format text prompt (Gemini chỉ nhận chuỗi)
            prompt_text = selected_prompt.format(
                context=context,
                question=question,
                language=language
            )

            # 3️⃣ System message cứng (bắt buộc tuân thủ ngôn ngữ)
            system_message = {
                "vi": (
                "Bạn là trợ lý kỹ thuật ảo của Arbin Instruments — công ty hàng đầu về thiết bị thử nghiệm pin.\n\n"
                "MỤC TIÊU:\n"
                "- Hỗ trợ khách hàng Việt Nam trong việc tìm hiểu sản phẩm, thông số kỹ thuật, hướng dẫn sử dụng và khắc phục sự cố của Arbin.\n"
                "- Giải thích ngắn gọn, rõ ràng, có dấu đầy đủ.\n"
                "- Nếu không có đủ thông tin, hãy nói rõ điều đó và gợi ý nơi người dùng có thể xem thêm (ví dụ: www.arbin.com hoặc email support@arbin.com).\n\n"
                "YÊU CẦU NGÔN NGỮ:\n"
                "- Trả lời 100% bằng TIẾNG VIỆT có dấu.\n"
                "- Không sử dụng tiếng Anh.\n"
                "- Giữ giọng điệu chuyên nghiệp, rõ ràng, tập trung vào nội dung kỹ thuật.\n"
                "- Không cần chào hỏi hoặc cảm ơn trong phần trả lời.\n\n"
            ),
            "en": (
                "You are Arbin Instruments' virtual technical assistant — a global leader in battery testing systems.\n\n"
                "GOAL:\n"
                "- Help users understand Arbin products, specifications, setup guides, and troubleshooting steps.\n"
                "- Provide clear, accurate, and concise explanations.\n"
                "- If documentation is incomplete, state that honestly and suggest where to find more (e.g., www.arbin.com or support@arbin.com).\n\n"
                "LANGUAGE REQUIREMENTS:\n"
                "- Respond 100% in ENGLISH.\n"
                "- Do NOT use Vietnamese.\n"
                "- Maintain a professional and concise tone."
                "- Avoid greetings or thank-you phrases at the beginning of responses.\n\n"
            )
            }.get(language, "You are Arbin assistant.\n\n")

            # 4️⃣ Gộp system + chat_history + prompt
            full_prompt = (
                f"{system_message}"
                f"Previous chat history:\n{chat_history}\n\n"
                f"{prompt_text}"
            )

            print("🧠 [Gemini Prompt Preview]:")
            print(full_prompt[:600] + "...\n")

            # 5️⃣ Gọi Gemini (tuỳ theo SDK)
            if hasattr(self.llm, "generate_content"):
                response = self.llm.generate_content(full_prompt)
                if hasattr(response, "text"):
                    response = response.text
            else:
                response = self.llm.invoke(full_prompt)
                if hasattr(response, "content"):
                    response = response.content

            # 6️⃣ Chuẩn hoá output
            response = self._validate_response_language(response, language)
            return response.strip()

        except Exception as e:
            print(f"❌ Lỗi generate_response (Gemini): {e}")
            import traceback; traceback.print_exc()
            return (
                "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời. "
                "Vui lòng thử lại hoặc liên hệ support@arbin.com."
            )



    def _validate_response_language(self, response: str, expected_language: str) -> str:
        """Validate ngôn ngữ của response và sửa nếu cần"""
        if not response:
            return response
        
        # Phát hiện ngôn ngữ
        vietnamese_chars = "àáạảãâầấậẩẫêềếệểễòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        has_vietnamese = any(char in response for char in vietnamese_chars)
        
        detected_language = "vi" if has_vietnamese else "en"
        
        if detected_language != expected_language:
            print(f"⚠️ WARNING: Language mismatch! Expected: {expected_language}, Got: {detected_language}")
            
            # Thêm warning dựa trên ngôn ngữ mong muốn
            warnings = {
                "vi": "[Lưu ý: Đây là bản dịch tự động từ tiếng Anh]\n\n",
                "en": "[Note: This is an auto-translation from Vietnamese]\n\n"
            }
            
            warning = warnings.get(expected_language, "")
            response = warning + response
        
        return response

    # ================= RETRIEVER IMPLEMENTATION =================
    def _get_retriever(self):
        """Tạo retriever từ vector store"""
        class VectorStoreRetriever:
            def __init__(self, vector_store, k=5):
                self.vector_store = vector_store
                self.k = k
            
            def get_relevant_documents(self, query):
                results = self.vector_store.search_similar(query, k=self.k)
                documents = []
                for result in results:
                    # Tạo document object tương thích với LangChain
                    class Document:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    doc = Document(
                        page_content=result['text'],
                        metadata=result['metadata']
                    )
                    # Thêm score nếu có
                    if 'score' in result:
                        doc.score = result['score']
                    documents.append(doc)
                return documents
        
        return VectorStoreRetriever(self.vector_store, k=5)

    # ================= HÀM XỬ LÝ CÂU HỎI CHÍNH =================
    def get_response(self, question: str, session_id: str = "default",
                 language: str = None) -> Dict[str, Any]:  # Đổi mặc định thành None
        """
        Xử lý câu hỏi với pipeline hoàn chỉnh: NLU → Retrieval → Generation
        """
        try:
             # === XỬ LÝ ĐẶC BIỆT CHO GREETING ===
            if self._is_greeting(question):
                print(f"🎯 Detected as greeting")
                detected_lang = self._detect_language(question)
                
                greeting_responses = {
                    "vi": "Xin chào! Tôi là trợ lý ảo của Arbin Instruments. Tôi có thể giúp gì cho bạn về sản phẩm và dịch vụ của Arbin?",
                    "en": "Hello! I'm Arbin Instruments virtual assistant. How can I help you with Arbin products and services today?"
                }
                
                response = greeting_responses.get(detected_lang, greeting_responses["en"])
                self.memory_manager.save_context(session_id, question, response, "greeting", {})
                
                return {
                    "answer": response,
                    "intent": "greeting",
                    "entities": {},
                    "sources": [],
                    "confidence": 1.0,
                    "has_context": False,
                    "language": detected_lang
                }
            # === BƯỚC 0: PHÁT HIỆN NGÔN NGỮ ===
            final_language = self._resolve_language(question, language)
            print(f"🌐 Final language for response: {final_language}")
            
            # === BƯỚC 1: Phân tích NLU với language đã detect ===
            nlu_result = self.nlu_processor.process_nlu(question, final_language, session_id)
            intent = nlu_result["intent"]
            entities = nlu_result["entities"]
            enriched_text = nlu_result.get("enriched_text", question)
            
            print(f"🔍 NLU Analysis: intent='{intent}', language={final_language}, confidence={nlu_result['overall_confidence']:.2f}")
            if entities:
                print(f"   Entities: { {k: v for k, v in entities.items() if v} }")

            # Bước 2: Lấy dữ liệu hội thoại từ bộ nhớ
            chat_history = self.memory_manager.get_chat_history(session_id)

            # Bước 3: Xử lý đặc biệt cho từng intent
            effective_query = self._enhance_query_for_retrieval(enriched_text, intent, entities)
            
            print(f"📝 Effective query for retrieval: '{effective_query}'")
            
            # Bước 4: Retrieve documents
            retriever = self._get_retriever()
            docs = retriever.get_relevant_documents(effective_query)
            
            if not docs:
                # Fallback: Thử với query gốc
                docs = retriever.get_relevant_documents(question)
                
            if not docs:
                response = self._handle_no_documents(intent, question, final_language, chat_history)
                self.memory_manager.save_context(session_id, question, response, intent, entities)
                return {
                    "answer": response,
                    "intent": intent,
                    "entities": entities,
                    "sources": [],
                    "confidence": nlu_result["overall_confidence"],
                    "has_context": False,
                    "language": final_language  # Thêm language vào response
                }

            # Bước 5: Format context từ documents
            context = self._format_context(docs, intent, entities)
            
            # Bước 6: Chọn prompt và generate response
            response = self._generate_response(
                question=question,
                context=context,
                intent=intent,
                language=final_language,  # Dùng final_language
                chat_history=chat_history,
                entities=entities
            )
            
            # Bước 7: Format sources
            sources = self._format_sources(docs)
            
            # Bước 8: Lưu vào memory
            self.memory_manager.save_context(session_id, question, response, intent, entities)
            
            return {
                "answer": response,
                "intent": intent,
                "entities": entities,
                "sources": sources,
                "confidence": nlu_result["overall_confidence"],
                "has_context": True,
                "language": final_language,  # Thêm language vào response
                "context_preview": context[:500] + "..." if len(context) > 500 else context
            }

        except Exception as e:
            print(f"❌ Lỗi trong get_response: {str(e)}")
            print(traceback.format_exc())
            
            self.memory_manager.save_context(session_id, question, "Lỗi hệ thống", "error", {})
            
            return {
                "answer": "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau hoặc liên hệ support@arbin.com.",
                "intent": "error",
                "entities": {},
                "sources": [],
                "confidence": 0.0,
                "has_context": False,
                "language": "vi"
            }

    # ================= HÀM HỖ TRỢ (giữ nguyên từ bản gốc, chỉ sửa lỗi nhỏ) =================
    
    def _enhance_query_for_retrieval(self, query: str, intent: str, entities: Dict) -> str:
        """Enhance query để cải thiện retrieval"""
        if not query:
            query = ""
        
        enhanced_parts = [query]
        
        # Thêm product names
        if entities and entities.get("product_names"):
            for product in entities["product_names"][:2]:
                if product:
                    enhanced_parts.append(str(product))
        
        # Thêm từ khóa dựa trên intent
        intent_keywords = {
            "product_inquiry": ["product", "model", "specifications"],
            "technical_support": ["error", "problem", "troubleshoot"],
            "specification_request": ["specification", "parameter", "technical"],
            "comparison_request": ["compare", "difference", "versus"],
            "application_info": ["application", "use", "purpose"],
            "pricing_inquiry": ["price", "cost", "quote"]
        }
        
        if intent in intent_keywords:
            enhanced_parts.extend(intent_keywords[intent])
        
        # Lọc và join
        filtered_parts = []
        for part in enhanced_parts:
            if part and str(part).strip():
                filtered_parts.append(str(part).strip())
        
        if not filtered_parts:
            return "general inquiry"
        
        # Remove duplicates
        unique_parts = []
        seen = set()
        for part in filtered_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        return " ".join(unique_parts)

    def _format_context(self, docs, intent: str, entities: Dict) -> str:
        """Format documents thành context string phù hợp với intent"""
        if not docs:
            return ""
        
        context_parts = []
        
        # Header dựa trên intent
        intent_headers = {
            "product_inquiry": "THÔNG TIN SẢN PHẨM ARBIN:",
            "technical_support": "TÀI LIỆU KỸ THUẬT ARBIN:",
            "specification_request": "THÔNG SỐ KỸ THUẬT ARBIN:",
            "comparison_request": "THÔNG TIN SO SÁNH SẢN PHẨM:",
            "application_info": "THÔNG TIN ỨNG DỤNG ARBIN:",
            "general_info": "THÔNG TIN ARBIN:"
        }
        
        header = intent_headers.get(intent, "THÔNG TIN THAM KHẢO:")
        context_parts.append(header)
        context_parts.append("")
        
        # Lọc và format documents
        for i, doc in enumerate(docs[:5]):  # Giới hạn 5 documents
            # Extract metadata
            metadata = getattr(doc, 'metadata', {})
            title = metadata.get('title', f"Document {i+1}")
            source = metadata.get('source', 'Unknown')
            
            # Format content (giới hạn độ dài)
            content = doc.page_content
            if len(content) > 800:
                content = content[:800] + "..."
            
            # Thêm vào context
            context_parts.append(f"[{i+1}] {title} ({source})")
            context_parts.append(content)
            context_parts.append("---")
        
        return "\n".join(context_parts)

    

    def _post_process_response(self, response: str, intent: str, entities: Dict) -> str:
        """Làm câu trả lời thân thiện - FIX NHANH"""
        if not isinstance(response, str):
            response = str(response)
        
        # 1. Phát hiện greeting trong response
        response_lower = response.lower()
        greeting_words = ["xin chào", "hello", "hi", "chào", "hey", "greetings"]
        
        is_greeting_response = any(
            word in response_lower and response_lower.count(word) >= 2 
            for word in greeting_words
        )
        
        # Nếu là greeting response, không thêm gì cả
        if is_greeting_response:
            # Chỉ giữ lại 1 lời chào
            for word in greeting_words:
                if word in response_lower:
                    # Đếm số lần xuất hiện
                    count = response_lower.count(word)
                    if count > 1:
                        # Thay thế tất cả trừ lần đầu tiên
                        parts = response.split(word)
                        if len(parts) > 2:
                            response = parts[0] + word + ''.join(parts[2:])
                    break
            
            return response.strip()
        
        # 2. Phát hiện ngôn ngữ
        vietnamese_chars = "àáạảãâầấậẩẫêềếệểễ"
        lang = "vi" if any(char in response for char in vietnamese_chars) else "en"
        
        # 3. KIỂM TRA: Nếu response đã bắt đầu bằng lời chào/cảm ơn, không thêm nữa
        starts_with_greeting = response_lower.startswith(
            ("cảm ơn", "thanks", "thank you", "xin chào", "hello", "hi", "hey")
        )
        
        if not starts_with_greeting:
            # Thêm prefix đơn giản
            simple_prefixes = {
                "vi": ["", ""],  # Không thêm prefix
                "en": ["", ""]
            }
            prefix = simple_prefixes.get(lang, [""])[0]
            response = prefix + response
        
        # 4. Thêm emoji đơn giản
        if lang == "vi":
            emoji = "💡 "
            response = emoji + response
        
        # 5. Clean up
        response = re.sub(r'\n{3,}', '\n\n', response)
        return response.strip()

    def _format_sources(self, docs) -> List[Dict[str, str]]:
        """Format sources cho hiển thị"""
        sources = []
        
        for i, doc in enumerate(docs[:3]):  # Chỉ lấy top 3 sources
            metadata = getattr(doc, 'metadata', {})
            
            source = {
                'index': i + 1,
                'title': metadata.get('title', f"Document {i+1}"),
                'source_type': metadata.get('source', 'Unknown'),
                'url': metadata.get('url', ''),
                'file_name': metadata.get('file_name', ''),
                'relevance_score': getattr(doc, 'score', 0) if hasattr(doc, 'score') else 0,
                'content_preview': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content
            }
            sources.append(source)
        
        return sources

    def _handle_no_documents(self, intent: str, question: str, language: str, chat_history: str) -> str:
        """Xử lý thân thiện khi không tìm thấy documents"""
        
        intent_responses = {
            "product_inquiry": {
                "en": "I couldn't find specific information about this product in our current documentation. Please visit www.arbin.com/products for detailed product information.",
                "vi": "Tôi không tìm thấy thông tin cụ thể về sản phẩm này trong tài liệu hiện có. Vui lòng truy cập www.arbin.com/products để biết thông tin chi tiết về sản phẩm."
            },
            "technical_support": {
                "en": "I couldn't find troubleshooting information for this specific issue. Please contact Arbin technical support at support@arbin.com for assistance.",
                "vi": "Tôi không tìm thấy thông tin xử lý sự cố cho vấn đề cụ thể này. Vui lòng liên hệ bộ phận hỗ trợ kỹ thuật Arbin tại support@arbin.com để được hỗ trợ."
            },
            "specification_request": {
                "en": "I couldn't find the requested specifications in our current documentation. Please check the product datasheets at www.arbin.com/resources.",
                "vi": "Tôi không tìm thấy thông số kỹ thuật được yêu cầu trong tài liệu hiện có. Vui lòng kiểm tra bảng dữ liệu sản phẩm tại www.arbin.com/resources."
            },
            "pricing_inquiry": {
                "en": "Pricing information is not available in our public documentation. Please contact sales@arbin.com for a customized quote.",
                "vi": "Thông tin giá cả không có sẵn trong tài liệu công khai. Vui lòng liên hệ sales@arbin.com để nhận báo giá tùy chỉnh."
            },
            "general_info": {
                "en": "I couldn't find relevant information in our documentation. Please try rephrasing your question or visit www.arbin.com for more information.",
                "vi": "Tôi không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thử diễn đạt lại câu hỏi hoặc truy cập www.arbin.com để biết thêm thông tin."
            },
            "comparison_request": {
                "en": "I couldn't find comparison information in our current documentation. Please check the product comparison guides at www.arbin.com/resources.",
                "vi": "Tôi không tìm thấy thông tin so sánh trong tài liệu hiện có. Vui lòng kiểm tra hướng dẫn so sánh sản phẩm tại www.arbin.com/resources."
            },
            "application_info": {
                "en": "I couldn't find application-specific information in our documentation. Please contact applications@arbin.com for specialized advice.",
                "vi": "Tôi không tìm thấy thông tin ứng dụng cụ thể trong tài liệu. Vui lòng liên hệ applications@arbin.com để được tư vấn chuyên sâu."
            }
        }
        # Thêm knowledge base về MITS Pro
        mits_pro_knowledge = {
            "vi": """
            **Về MITS Pro:**
            MITS Pro là phần mềm điều khiển chính của Arbin, thường cung cấp:
            1. Giao diện lập trình menu cho chu kỳ sạc/xả
            2. Các điều kiện dừng (stop conditions) có thể cấu hình
            3. Logic điều khiển linh hoạt
            
            **Gợi ý cấu hình:**
            Để thiết lập điều kiện dừng khi điện áp đạt V và dòng ≤ I:
            - Tìm menu "Stop Conditions" hoặc "Safety Limits"
            - Đặt Voltage Stop = giá trị V mong muốn
            - Đặt Current Stop ≤ giá trị I mong muốn
            - Kích hoạt cả hai điều kiện đồng thời
            
            **Tài nguyên hỗ trợ:**
            - User Manual MITS Pro (tìm trong Documents/Software)
            - Video tutorials trên YouTube chanel Arbin
            - Email hỗ trợ: support@arbin.com
            """,
            
            "en": """
            **About MITS Pro:**
            MITS Pro is Arbin's main control software, typically providing:
            1. Menu-driven programming interface for charge/discharge cycles
            2. Configurable stop conditions
            3. Flexible control logic
            
            **Configuration tips:**
            To set stop condition when voltage reaches V and current ≤ I:
            - Look for "Stop Conditions" or "Safety Limits" menu
            - Set Voltage Stop = desired V value
            - Set Current Stop ≤ desired I value
            - Enable both conditions simultaneously
            
            **Support resources:**
            - MITS Pro User Manual (check Documents/Software folder)
            - Video tutorials on Arbin YouTube channel
            - Email support: support@arbin.com
            """
        }
        
        # Thêm thông tin liên hệ tổng hợp
        contact_knowledge = {
            "vi": """
            **Thông tin liên hệ Arbin Instruments:**
            
            📧 **Email hỗ trợ kỹ thuật:** support@arbin.com
            📧 **Email bán hàng:** sales@arbin.com
            🌐 **Website:** www.arbin.com
            
            **Giờ làm việc hỗ trợ (giờ Mỹ CST):**
            - Thứ 2 - Thứ 6: 8:00 AM - 5:00 PM
            - Có hỗ trợ khẩn cấp ngoài giờ
            
            **Để được hỗ trợ nhanh nhất:**
            1. Cung cấp số serial thiết bị (nếu có)
            2. Mô tả chi tiết vấn đề
            3. Đính kèm file log (nếu có lỗi)
            
            Nếu cần số điện thoại cụ thể, vui lòng kiểm tra trang "Contact Us" trên website.
            """,
            
            "en": """
            **Arbin Instruments Contact Information:**
            
            📧 **Technical Support:** support@arbin.com
            📧 **Sales Inquiries:** sales@arbin.com
            🌐 **Website:** www.arbin.com
            
            **Support Hours (US CST):**
            - Monday - Friday: 8:00 AM - 5:00 PM
            - Emergency support available
            
            **For fastest support:**
            1. Provide equipment serial number (if available)
            2. Describe the issue in detail
            3. Attach log files (if error occurs)
            
            For specific phone numbers, please check the "Contact Us" page on the website.
            """
        }
        
        # Xác định ngôn ngữ
        lang = language if language in ["vi", "en"] else "en"
        
        # Kiểm tra nội dung câu hỏi
        question_lower = question.lower()
        
        # Thêm prefix thân thiện
        friendly_intro = {
            "vi": "Cảm ơn bạn đã hỏi! ",
            "en": "Thanks for asking! "
        }.get(lang, "")
        
        # Xử lý từng trường hợp
        if "mits pro" in question_lower or any(word in question_lower for word in ["cấu hình", "thiết lập", "setup", "configure"]):
            response = f"{friendly_intro}Tôi hiểu bạn cần hướng dẫn cấu hình MITS Pro.\n\n{mits_pro_knowledge.get(lang, mits_pro_knowledge['en'])}"
        
        elif any(word in question_lower for word in ["số điện thoại", "phone", "liên hệ", "contact", "địa chỉ"]):
            response = f"{friendly_intro}{contact_knowledge.get(lang, contact_knowledge['en'])}"
        
        elif intent in intent_responses:
            base_response = intent_responses[intent].get(lang, intent_responses[intent]["en"])
            response = f"{friendly_intro}{base_response}"
        
        else:
            fallback = {
                "vi": "Tôi chưa tìm thấy thông tin cụ thể trong tài liệu hiện có. Bạn có thể thử diễn đạt lại câu hỏi hoặc truy cập www.arbin.com để tìm thêm thông tin.",
                "en": "I haven't found specific information in our current documentation. You might try rephrasing your question or visit www.arbin.com for more information."
            }.get(lang, fallback["en"])
            response = f"{friendly_intro}{fallback}"
        
        # Thêm đề xuất tiếp theo
        next_step = {
            "vi": "\n\nTôi có thể giúp gì thêm cho bạn không?",
            "en": "\n\nIs there anything else I can help you with?"
        }.get(lang, "")
        
        return response + next_step

    # ================= BATCH PROCESSING =================
    def batch_get_response(self, questions: List[str], session_id: str = "default",
                          language: str = "en") -> List[Dict[str, Any]]:
        """Xử lý hàng loạt câu hỏi"""
        results = []
        for question in questions:
            try:
                result = self.get_response(question, session_id, language)
                results.append(result)
            except Exception as e:
                print(f"❌ Lỗi xử lý câu hỏi '{question}': {e}")
                results.append({
                    "answer": "Error processing question",
                    "intent": "error",
                    "entities": {},
                    "sources": [],
                    "confidence": 0.0,
                    "has_context": False
                })
        
        return results

    # ================= SYSTEM STATUS =================
    def get_system_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hệ thống"""
        try:
            # Kiểm tra vector store
            store_status = "Unknown"
            try:
                stats = self.vector_store.get_collection_stats()
                count = stats.get('total_documents', 0)
                store_status = f"Operational ({count} documents)"
            except Exception as e:
                store_status = f"Not accessible: {str(e)}"
            
            # Kiểm tra memory
            memory_status = "Operational" if self.memory_manager else "Not initialized"
            
            # Kiểm tra LLM
            llm_status = "Operational" if self.llm else "Not initialized"
            
            return {
                'status': 'operational',
                'components': {
                    'vector_store': store_status,
                    'memory_manager': memory_status,
                    'llm': llm_status,
                    'nlu_processor': 'Operational'
                },
                'configuration': {
                    'retrieval_k': 5,
                    'language_default': 'en',
                    'max_context_length': 4000
                }
            }
            
        except Exception as e:
            return {
                'status': f'error: {str(e)}',
                'components': {
                    'vector_store': 'Unknown',
                    'memory_manager': 'Unknown',
                    'llm': 'Unknown',
                    'nlu_processor': 'Unknown'
                }
            }


# Factory function
def create_arbin_retrieval_qa(llm, vector_store) -> ArbinRetrievalQA:
    """Factory function để tạo ArbinRetrievalQA"""
    return ArbinRetrievalQA(llm=llm, vector_store=vector_store)