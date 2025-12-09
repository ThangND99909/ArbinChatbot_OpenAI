# ============================================================
# NLU PROCESSOR CHO ARBIN INSTRUMENTS
# ------------------------------------------------------------
# Tác dụng:
#  - Phân tích câu hỏi người dùng để xác định "intent" (ý định)
#  - Trích xuất "entities" (thực thể như tên sản phẩm, thông số, lỗi, v.v.)
#  - Hỗ trợ ngữ cảnh hội thoại (Context Memory) để duy trì mạch trò chuyện
#  - Kết hợp AI (Gemini) + keyword fallback để đảm bảo ổn định
# ============================================================

import json
import re
import logging
import unicodedata
from typing import Dict, Any, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from ai_core.llm_chain import GeminiLLM
from .prompts import intent_prompt, entity_prompt
from .parsers import NLUOutputParser
import traceback

# ========================
# Logging setup
# ========================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ============================================================
# BỘ NHỚ NGỮ CẢNH ĐƠN GIẢN (ContextMemory)
# ============================================================
class ContextMemory:
    """
    Lưu lại intent và entities gần nhất mà người dùng đã nói.
    Dùng để giúp chatbot hiểu mạch hội thoại và duy trì ngữ cảnh.
    """
    def __init__(self):
        self.last_intent = None
        self.last_entities = {}
        self.last_product_mentioned = None  # Sản phẩm vừa được nhắc đến trong hội thoại

    def update(self, intent: str, entities: Dict[str, Any]):
        """Cập nhật intent và entities mới nhất"""
        if intent and intent != "unknown":
            self.last_intent = intent
        
        if entities:
            self.last_entities = entities
            # Nếu có sản phẩm được nhắc đến, lưu lại
            if entities.get("product_names"):
                self.last_product_mentioned = entities["product_names"][0]

    def get_context(self) -> Dict[str, Any]:
        """Trả về ngữ cảnh hiện tại gồm intent, entity, sản phẩm"""
        return {
            "last_intent": self.last_intent,
            "last_entities": self.last_entities,
            "last_product_mentioned": self.last_product_mentioned
        }

# ============================================================
# HÀM HỖ TRỢ (TIỀN XỬ LÝ)
# ============================================================
def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp keyword dễ hơn"""
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _contains_any(haystack: str, needles) -> bool:
    """Kiểm tra xem chuỗi có chứa bất kỳ từ khóa nào trong danh sách không"""
    return any(n in haystack for n in needles)

# ============================================================
# DANH SÁCH TỪ KHÓA CHUYÊN BIỆT CHO ARBIN
# ============================================================
# Mục đích: giúp nhận dạng intent và entity mà không cần AI (fallback mode)

# Các cụm thường xuất hiện trong câu hỏi
QUESTION_TRIGGERS = [
    "là gì", "la gi", "giới thiệu", "gioi thieu", "thông tin", "thong tin",
    "bao nhiêu", "bao nhieu", "vì sao", "vi sao", "?", "how", "what", "which",
    "tại sao", "tai sao", "thế nào", "the nao", "cách", "cach", "how to"
]

# Từ khóa sản phẩm Arbin (ví dụ BT2000, MITS, EV Test...)
PRODUCT_KEYWORDS = [
    "bt", "bt-", "lbt", "mbt", "mits", "mits pro", "ev test", "battery tester",
    "cell tester", "battery cycler", "test system", "arbin", "testing system",
    "hardware", "software", "win", "daq", "windaq", "console", "client"
]

# Thuật ngữ kỹ thuật (technical keywords)
TECHNICAL_KEYWORDS = [
    "voltage", "current", "capacity", "power", "impedance", "resistance",
    "soc", "soh", "cycle", "charging", "discharging", "calibration",
    "accuracy", "resolution", "range", "channel", "frequency", "temperature",
    "measurement", "testing", "analysis", "data acquisition", "monitoring"
]

# Từ khóa liên quan hỗ trợ kỹ thuật
TECH_SUPPORT_KEYWORDS = [
    "error", "problem", "issue", "bug", "fix", "repair", "troubleshoot",
    "help", "how to", "why", "not working", "lỗi", "sự cố", "không hoạt động",
    "hướng dẫn", "cách sử dụng", "giải quyết", "support", "assistance",
    "crash", "fail", "broken", "malfunction", "calibration error"
]

# Từ khóa về thông số kỹ thuật
SPECIFICATION_KEYWORDS = [
    "spec", "specification", "parameter", "technical data", "feature",
    "capacity", "voltage", "current", "power", "accuracy", "resolution",
    "thông số", "đặc tính", "tính năng", "dải đo", "độ chính xác", "range"
]

# Từ khóa giá cả
PRICING_KEYWORDS = [
    "price", "cost", "quote", "quotation", "budget", "expensive", "cheap",
    "how much", "giá", "chi phí", "báo giá", "định giá", "kinh phí",
    "affordable", "pricing", "estimate", "quotation"
]

# Từ khóa so sánh sản phẩm
COMPARISON_KEYWORDS = [
    "compare", "versus", "vs", "difference", "different", "better",
    "best", "worst", "advantages", "disadvantages", "pros", "cons",
    "so sánh", "khác biệt", "ưu điểm", "nhược điểm", "tốt hơn", "hơn kém"
]

# Từ khóa về ứng dụng
APPLICATION_KEYWORDS = [
    "application", "use", "usage", "purpose", "suitable for", "scenario",
    "ứng dụng", "sử dụng", "mục đích", "phù hợp", "tình huống", "lĩnh vực",
    "industry", "field", "purpose", "for what", "dùng cho"
]

# Địa điểm ứng dụng (lab, nhà máy,...)
LOCATION_TYPES = [
    "laboratory", "lab", "factory", "manufacturing", "research center",
    "university", "college", "test facility", "quality control",
    "production line", "phòng thí nghiệm", "nhà máy", "xưởng sản xuất",
    "trung tâm nghiên cứu", "đại học", "cơ sở thử nghiệm"
]

# ============================================================
# 🧩 NLU PROCESSOR CHÍNH CHO ARBIN
# ============================================================
class NLUProcessor:
    """
    Xử lý toàn bộ pipeline NLU:
    - Intent detection (phát hiện ý định)
    - Entity extraction (trích xuất thực thể)
    - Context tracking (theo dõi ngữ cảnh hội thoại)
    """

    def __init__(self, llm=None, memory_manager=None):
        """
        llm: mô hình LLM (Gemini)
        memory_manager: đối tượng quản lý hội thoại (để lấy intent, question, v.v.)
        """
        self.llm = llm or GeminiLLM()
        self.memory_manager = memory_manager
        self.memory = ContextMemory()  # Bộ nhớ ngữ cảnh cục bộ
        
        # Tạo chain cho Intent Detection
        self.intent_chain = LLMChain(
            llm=self.llm,
            prompt=intent_prompt,
            output_parser=NLUOutputParser(),
            output_key="answer"
        )
        
        # Tạo chain cho Entity Extraction
        self.entity_chain = LLMChain(
            llm=self.llm,
            prompt=entity_prompt,
            output_parser=NLUOutputParser(),
            output_key="answer"
        )
        
        logger.info("✅ Arbin NLUProcessor initialized")

    # =======================================================
    # HÀM LẤY TIN NHẮN ASSISTANT GẦN NHẤT
    # =======================================================
    def _get_last_assistant_message(self, session_id: str) -> str:
        """Truy xuất phản hồi gần nhất từ chatbot (dùng cho context linking)"""
        if not self.memory_manager:
            return ""
        try:
            msgs = self.memory_manager.get_messages(session_id)
            for m in reversed(msgs):
                if hasattr(m, "type") and m.type != "human":
                    return getattr(m, "content", "") or ""
        except Exception as e:
            logger.debug(f"Không lấy được last assistant message: {e}")
        return ""

    # =======================================================
    # PHÁT HIỆN INTENT (Detect Intent)
    # =======================================================
    def detect_intent(self, question: str, language: str = "en", session_id: str = "default") -> Dict[str, Any]:
        """
        Dự đoán ý định của người dùng cho Arbin Instruments.
        Kết hợp AI (Gemini) + heuristic (keyword fallback)
        """
        try:
            # Gọi LLM chain để dự đoán intent
            try:
                raw = self.intent_chain.invoke({"question": question, "language": language})
            except Exception:
                # Nếu lỗi LLM → fallback sang keyword
                return self._get_intent_by_keywords_fallback(question, session_id)
            
            # Chuẩn hóa output
            if isinstance(raw, dict):
                if 'answer' in raw:
                    output_text = raw['answer']
                elif 'text' in raw:
                    output_text = raw['text']
                else:
                    output_text = str(raw)
            else:
                output_text = str(raw)
            
            # Parse kết quả từ LLM (dạng JSON)
            try:
                if isinstance(output_text, dict):
                    output_text = json.dumps(output_text, ensure_ascii=False)
                parsed = self.intent_chain.output_parser.parse(output_text)
                if isinstance(parsed, dict):
                    intent = parsed.get("intent", "unknown")
                    confidence = float(parsed.get("confidence", 0.0))
                else:
                    intent = "unknown"
                    confidence = 0.0
            except Exception:
                intent = "unknown"
                confidence = 0.0
            
            # Lấy intent và question gần nhất từ memory_manager
            last_intent, last_question = "", ""
            if self.memory_manager:
                last_intent = self.memory_manager.get_last_intent(session_id)
                last_question = self.memory_manager.get_last_question(session_id)

            # Chuẩn hóa text để match keywords
            t_lc = question.strip().lower()
            t_ascii = _strip_accents(t_lc)
            words = t_ascii.split()
            is_short = len(words) <= 4  # Câu ngắn có thể cần enrichment

            # Kiểm tra keyword để tăng độ tin cậy
            has_tech_support_kw = _contains_any(t_lc, TECH_SUPPORT_KEYWORDS) or _contains_any(t_ascii, TECH_SUPPORT_KEYWORDS)
            has_spec_kw = _contains_any(t_lc, SPECIFICATION_KEYWORDS) or _contains_any(t_ascii, SPECIFICATION_KEYWORDS)
            has_pricing_kw = _contains_any(t_lc, PRICING_KEYWORDS) or _contains_any(t_ascii, PRICING_KEYWORDS)
            has_comparison_kw = _contains_any(t_lc, COMPARISON_KEYWORDS) or _contains_any(t_ascii, COMPARISON_KEYWORDS)
            has_application_kw = _contains_any(t_lc, APPLICATION_KEYWORDS) or _contains_any(t_ascii, APPLICATION_KEYWORDS)
            has_product_kw = _contains_any(t_lc, PRODUCT_KEYWORDS) or _contains_any(t_ascii, PRODUCT_KEYWORDS)
            has_question_word = _contains_any(t_lc, QUESTION_TRIGGERS) or _contains_any(t_ascii, QUESTION_TRIGGERS)

            # =======================================================
            # Nâng cấp intent dựa trên heuristic / keyword
            # =======================================================
            enriched_text = None
            
            # Nếu intent unknown hoặc confidence thấp → fallback theo từ khóa
            if intent == "unknown" or confidence < 0.3:
                if has_tech_support_kw:
                    intent = "technical_support"
                    confidence = max(confidence, 0.7)
                elif has_spec_kw:
                    intent = "specification_request"
                    confidence = max(confidence, 0.7)
                elif has_pricing_kw:
                    intent = "pricing_inquiry"
                    confidence = max(confidence, 0.7)
                elif has_comparison_kw:
                    intent = "comparison_request"
                    confidence = max(confidence, 0.7)
                elif has_application_kw:
                    intent = "application_info"
                    confidence = max(confidence, 0.7)
                elif has_product_kw and has_question_word:
                    intent = "product_inquiry"
                    confidence = max(confidence, 0.6)
                elif has_product_kw:
                    intent = "product_inquiry"
                    confidence = max(confidence, 0.5)

            # Nếu câu ngắn, gắn thêm context trước đó để hiểu hơn
            if last_intent == "product_inquiry" and is_short and has_product_kw:
                intent = "product_inquiry"
                enriched_text = f"{last_question} {question}" if last_question else question
            
            elif last_intent == "product_inquiry" and has_spec_kw:
                intent = "specification_request"
                enriched_text = f"specifications of {self.memory.last_product_mentioned or 'the product'} {question}"

            if not enriched_text and is_short and last_question:
                enriched_text = f"{last_question} {question}"
            if enriched_text is None:
                enriched_text = question

            # Cập nhật bộ nhớ tạm (context memory)
            if intent != "unknown":
                self.memory.update(intent, {})

            return {
                "intent": intent,
                "confidence": confidence,
                "last_intent": last_intent,
                "last_question": last_question,
                "last_product_mentioned": self.memory.last_product_mentioned,
                "enriched_text": enriched_text,
                "keywords_detected": {
                    "has_product": has_product_kw,
                    "has_technical": _contains_any(t_lc, TECHNICAL_KEYWORDS),
                    "has_specification": has_spec_kw,
                    "has_support": has_tech_support_kw,
                    "has_pricing": has_pricing_kw,
                    "has_comparison": has_comparison_kw,
                    "has_application": has_application_kw
                }
            }

        except Exception:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "last_intent": "",
                "last_question": "",
                "enriched_text": None,
                "keywords_detected": {}
            }

    # =======================================================
    # FALLBACK PHÁT HIỆN INTENT BẰNG KEYWORDS (KHÔNG CẦN AI)
    # =======================================================
    def _get_intent_by_keywords_fallback(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Khi Gemini API bị lỗi hoặc timeout, hàm này sẽ
        tự động xác định intent chỉ dựa trên từ khóa.
        """
        logger.warning("Using keyword-only fallback intent detection")
        
        # Chuẩn hóa text (chữ thường + bỏ dấu)
        t_lc = question.strip().lower()
        t_ascii = _strip_accents(t_lc)
        
        # Kiểm tra từng nhóm keyword
        has_product_kw = _contains_any(t_lc, PRODUCT_KEYWORDS) or _contains_any(t_ascii, PRODUCT_KEYWORDS)
        has_tech_support_kw = _contains_any(t_lc, TECH_SUPPORT_KEYWORDS) or _contains_any(t_ascii, TECH_SUPPORT_KEYWORDS)
        has_spec_kw = _contains_any(t_lc, SPECIFICATION_KEYWORDS) or _contains_any(t_ascii, SPECIFICATION_KEYWORDS)
        has_pricing_kw = _contains_any(t_lc, PRICING_KEYWORDS) or _contains_any(t_ascii, PRICING_KEYWORDS)
        has_comparison_kw = _contains_any(t_lc, COMPARISON_KEYWORDS) or _contains_any(t_ascii, COMPARISON_KEYWORDS)
        has_application_kw = _contains_any(t_lc, APPLICATION_KEYWORDS) or _contains_any(t_ascii, APPLICATION_KEYWORDS)
        has_question_word = _contains_any(t_lc, QUESTION_TRIGGERS) or _contains_any(t_ascii, QUESTION_TRIGGERS)
        
        # Mặc định intent chưa xác định
        intent = "unknown"
        confidence = 0.5  # confidence trung bình cho rule-based
        
        # Phân loại intent dựa trên nhóm từ khóa
        if has_tech_support_kw:
            intent = "technical_support"
            confidence = 0.7
        elif has_spec_kw:
            intent = "specification_request"
            confidence = 0.7
        elif has_pricing_kw:
            intent = "pricing_inquiry"
            confidence = 0.7
        elif has_comparison_kw:
            intent = "comparison_request"
            confidence = 0.7
        elif has_application_kw:
            intent = "application_info"
            confidence = 0.7
        elif has_product_kw and has_question_word:
            intent = "product_inquiry"
            confidence = 0.6
        elif has_product_kw:
            intent = "product_inquiry"
            confidence = 0.5
        
        # Lấy thông tin ngữ cảnh trước đó từ memory_manager (nếu có)
        last_intent, last_question = "", ""
        if self.memory_manager:
            last_intent = self.memory_manager.get_last_intent(session_id)
            last_question = self.memory_manager.get_last_question(session_id)
        
        # Nếu phát hiện có tên sản phẩm, trích xuất ra
        product_names = self._extract_product_names_from_text(question) if has_product_kw else []
        
        # Trả kết quả fallback
        return {
            "intent": intent,
            "confidence": confidence,
            "last_intent": last_intent,
            "last_question": last_question,
            "last_product_mentioned": product_names[0] if product_names else None,
            "enriched_text": question,
            "keywords_detected": {
                "has_product": has_product_kw,
                "has_technical": _contains_any(t_lc, TECHNICAL_KEYWORDS),
                "has_specification": has_spec_kw,
                "has_support": has_tech_support_kw,
                "has_pricing": has_pricing_kw,
                "has_comparison": has_comparison_kw,
                "has_application": has_application_kw,
                "has_question_word": has_question_word
            },
            "intent_override_applied": True  # đánh dấu dùng rule-based
        }

    # =======================================================
    # TRÍCH XUẤT TÊN SẢN PHẨM (Product Extraction)
    # =======================================================
    def _extract_product_names_from_text(self, text: str) -> List[str]:
        """
        Dò tìm tên sản phẩm Arbin trong câu hỏi
        Ví dụ: "BT2000", "MITS Pro", "Battery Tester"
        """
        products = []
        text_lower = text.lower()
        
        # Các pattern regex phổ biến cho tên sản phẩm Arbin
        product_patterns = [
            r'bt[-\s]?\d+',          # BT-2000, BT 2000
            r'lbt[-\s]?\d*',         # LBT series
            r'mbt[-\s]?\d*',         # MBT series
            r'mits\s*(?:pro|x)?',    # MITS Pro, MITS X
            r'ev\s*(?:test|testing)?',  # EV test
            r'battery\s+tester',     # battery tester
            r'cell\s+tester',        # cell tester
            r'battery\s+cycler',     # battery cycler
        ]
        
        # Dò từng pattern
        for pattern in product_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            products.extend(matches)
        
        return products

    # =======================================================
    # TRÍCH XUẤT THỰC THỂ (Entity Extraction)
    # =======================================================
    def extract_entities(self, question: str, language: str = "en") -> Dict[str, Any]:
        """
        Gọi LLM để phân tích câu hỏi và trích xuất các thực thể
        (ví dụ: tên sản phẩm, thông số, lỗi, phần mềm, v.v.)
        """
        try:
            # Thử gọi LLM chain
            try:
                result = self.entity_chain.invoke({"question": question, "language": language})
            except Exception:
                # Nếu LLM lỗi → fallback keyword extraction
                return {
                    "entities": self._keyword_entity_extraction(question),
                    "confidence": 0.4,
                    "raw_output": ""
                }
            
            # Chuẩn hóa output từ LLM
            if isinstance(result, dict):
                if 'answer' in result:
                    output_text = result['answer']
                elif 'text' in result:
                    output_text = result['text']
                else:
                    output_text = str(result)
            else:
                output_text = str(result)
            
            # Parse dữ liệu JSON trả về từ LLM
            try:
                if isinstance(output_text, dict):
                    output_text = json.dumps(output_text, ensure_ascii=False)
                
                parsed = self.entity_chain.output_parser.parse(output_text)
                
                if isinstance(parsed, dict):
                    # Nếu parse thành công
                    entities = parsed.get("entities", {
                        "product_names": [],
                        "technical_terms": [],
                        "specifications": [],
                        "applications": [],
                        "features": [],
                        "issues": [],
                        "software_components": [],
                        "locations": []
                    })
                    confidence = parsed.get("confidence", 0.7)
                else:
                    # Nếu parse lỗi, trả kết quả rỗng
                    entities = {k: [] for k in [
                        "product_names", "technical_terms", "specifications", "applications",
                        "features", "issues", "software_components", "locations"
                    ]}
                    confidence = 0.4
                    
            except Exception:
                # Nếu lỗi parse → fallback keyword
                entities = self._keyword_entity_extraction(question)
                confidence = 0.4
            
            # Kết hợp thêm các thực thể phát hiện bằng keyword
            keyword_entities = self._keyword_entity_extraction(question)
            for key, value in keyword_entities.items():
                if value and key in entities:
                    existing_values = set(entities[key])
                    for v in value:
                        if v not in existing_values:
                            entities[key].append(v)
            
            # Lưu lại entities vào context memory
            if any(entities.values()):
                self.memory.last_entities = entities
            
            return {
                "entities": entities,
                "confidence": confidence,
                "raw_output": str(output_text)[:500] if output_text else ""
            }
            
        except Exception:
            return {
                "entities": self._keyword_entity_extraction(question),
                "confidence": 0.4,
                "raw_output": ""
            }

    # =======================================================
    # TRÍCH XUẤT ENTITY BẰNG KEYWORDS (FALLBACK)
    # =======================================================
    def _keyword_entity_extraction(self, question: str) -> Dict[str, List[str]]:
        """
        Khi không có phản hồi từ AI → tự trích xuất thực thể bằng regex & từ khóa.
        Phù hợp với Arbin: nhận dạng thông số kỹ thuật, sản phẩm, phần mềm, lỗi,...
        """
        question_lower = question.lower()
        question_no_accents = _strip_accents(question_lower)
        
        entities = {
            "product_names": [],
            "technical_terms": [],
            "specifications": [],
            "applications": [],
            "features": [],
            "issues": [],
            "software_components": [],
            "locations": []
        }
        
        # Dò pattern sản phẩm
        product_patterns = [
            r'bt[-\s]?\d+', r'lbt[-\s]?\d*', r'mbt[-\s]?\d*', 
            r'mits\s*(?:pro|x)?', r'ev\s*(?:test|testing)?',
            r'battery\s+tester', r'cell\s+tester', r'battery\s+cycler'
        ]
        for pattern in product_patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            entities["product_names"].extend(matches)
        
        # Dò thuật ngữ kỹ thuật
        for term in TECHNICAL_KEYWORDS:
            if term in question_lower or term in question_no_accents:
                entities["technical_terms"].append(term)
        
        # Dò thông số kỹ thuật (dạng số + đơn vị)
        spec_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(v|volts?)', 'voltage'),
            (r'(\d+(?:\.\d+)?)\s*(a|amps?)', 'current'),
            (r'(\d+(?:\.\d+)?)\s*(w|watts?)', 'power'),
            (r'(\d+(?:\.\d+)?)\s*(ah|mah)', 'capacity'),
            (r'(\d+(?:\.\d+)?)\s*%', 'accuracy'),
            (r'(\d+)\s*(channel|ch)s?', 'channels'),
            (r'±?\s*(\d+(?:\.\d+)?)\s*(?:v|a|w|%)', 'spec_value')
        ]
        for pattern, spec_type in spec_patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            for match in matches:
                value = match[0] if isinstance(match, tuple) else match
                entities["specifications"].append(f"{value} {spec_type}")
        
        # Phần mềm Arbin (Windaq, Console, MITS,...)
        software_keywords = ['windaq', 'mits', 'console', 'client', 'server', 'software', 'interface']
        for keyword in software_keywords:
            if keyword in question_lower:
                entities["software_components"].append(keyword)
        
        # Địa điểm ứng dụng (lab, factory,...)
        for location in LOCATION_TYPES:
            if location in question_lower:
                entities["locations"].append(location)
        
        # Lỗi kỹ thuật hoặc sự cố
        issue_keywords = ['error', 'problem', 'issue', 'bug', 'fail', 'crash', 'not working', 'lỗi']
        for keyword in issue_keywords:
            if keyword in question_lower:
                entities["issues"].append(keyword)
        
        return {k: v for k, v in entities.items() if v}

    # =======================================================
    # PIPELINE TỔNG HỢP: process_nlu()
    # =======================================================
    def process_nlu(self, question: str, language: str = "en", session_id: str = "default") -> Dict[str, Any]:
        """
        Gọi toàn bộ pipeline NLU gồm:
        1. detect_intent() → xác định intent
        2. extract_entities() → trích xuất entities
        3. hợp nhất kết quả + thêm context
        """
        logger.info(f"Processing NLU for question: '{question[:50]}...'")
        
        effective_language = language if language else "en"
        
        # Chạy tuần tự 2 bước
        intent_result = self.detect_intent(question, effective_language, session_id)
        entity_result = self.extract_entities(question, effective_language)

        # Hợp nhất kết quả
        merged = {
            "query": question,
            "language": language,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "entities": entity_result["entities"],
            "entity_confidence": entity_result["confidence"],
            "context": {
                "last_intent": intent_result.get("last_intent", ""),
                "last_question": intent_result.get("last_question", ""),
                "last_product_mentioned": intent_result.get("last_product_mentioned"),
                "memory_context": self.memory.get_context()
            },
            "enriched_text": intent_result.get("enriched_text"),
            "keywords_detected": intent_result.get("keywords_detected", {}),
            "raw_outputs": {
                "intent_raw": intent_result,
                "entity_raw": entity_result.get("raw_output", "")
            }
        }
        
        # Tính điểm confidence tổng thể (weighted average)
        merged["overall_confidence"] = (
            intent_result["confidence"] * 0.6 + 
            entity_result["confidence"] * 0.4
        )
        
        # Gợi ý câu hỏi tiếp theo
        merged["suggested_responses"] = self._generate_suggested_responses(
            intent_result["intent"],
            entity_result["entities"]
        )
        
        logger.info(f"NLU Analysis complete: intent={merged['intent']}, confidence={merged['overall_confidence']:.2f}")
        return merged

    # =======================================================
    # TẠO CÂU GỢI Ý (Suggested Responses)
    # =======================================================
    def _generate_suggested_responses(self, intent: str, entities: Dict[str, List[str]]) -> List[str]:
        """
        Sinh ra các câu hỏi gợi ý cho người dùng,
        tùy theo intent hiện tại và thông tin đã phát hiện được.
        """
        suggestions = []
        
        # Intent: hỏi sản phẩm
        if intent == "product_inquiry":
            if entities.get("product_names"):
                product = entities["product_names"][0]
                suggestions = [
                    f"What are the key specifications of {product}?",
                    f"What applications is {product} best suited for?",
                    f"How does {product} compare to similar models?",
                    f"What is the price range for {product}?"
                ]
            else:
                suggestions = [
                    "Which Arbin product are you interested in?",
                    "Are you looking for battery test systems or software?",
                    "What capacity range do you need for your testing?"
                ]
        
        # Intent: hỗ trợ kỹ thuật
        elif intent == "technical_support":
            suggestions = [
                "What specific error message are you seeing?",
                "Which software version are you currently using?",
                "Have you checked the troubleshooting guide in the manual?",
                "Is this a hardware or software issue?"
            ]
        
        # Intent: hỏi thông số kỹ thuật
        elif intent == "specification_request":
            if entities.get("product_names"):
                product = entities["product_names"][0]
                suggestions = [
                    f"What is the voltage range of {product}?",
                    f"How many channels does {product} support?",
                    f"What is the measurement accuracy of {product}?"
                ]
            else:
                suggestions = [
                    "Which product specifications are you interested in?",
                    "Are you looking for voltage, current, or power specifications?",
                    "Do you need accuracy specifications or measurement ranges?"
                ]
        
        # Intent: so sánh sản phẩm
        elif intent == "comparison_request":
            if len(entities.get("product_names", [])) >= 2:
                products = " and ".join(entities["product_names"][:2])
                suggestions = [f"What specific aspects of {products} would you like to compare?"]
            else:
                suggestions = [
                    "Which products would you like to compare?",
                    "Are you comparing different series or models?",
                    "What criteria are important for your comparison?"
                ]
        
        # Intent: hỏi giá
        elif intent == "pricing_inquiry":
            suggestions = [
                "Are you looking for academic or commercial pricing?",
                "Do you need a formal quote or just a price range?",
                "Would you like information about leasing options?"
            ]
        
        # Intent: hỏi ứng dụng
        elif intent == "application_info":
            suggestions = [
                "What type of batteries are you testing?",
                "Is this for research, quality control, or production?",
                "What is your testing throughput requirement?"
            ]
        
        return suggestions[:3]  # chỉ lấy 3 câu đầu tiên

    # =======================================================
    # BATCH ANALYZE (PHÂN TÍCH NHIỀU CÂU)
    # =======================================================
    def batch_analyze(self, queries: List[str], language: str = "en") -> List[Dict[str, Any]]:
        """Phân tích hàng loạt câu hỏi → dùng trong testing hoặc huấn luyện."""
        results = []
        for query in queries:
            try:
                analysis = self.process_nlu(query, language)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing query '{query}': {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "intent": "error",
                    "confidence": 0.0
                })
        return results


# =======================================================
# FACTORY FUNCTION: TẠO NLU PROCESSOR CHO ARBIN
# =======================================================
def create_nlu_processor(llm=None, memory_manager=None) -> NLUProcessor:
    """Factory function tiện lợi để khởi tạo NLUProcessor"""
    return NLUProcessor(llm=llm, memory_manager=memory_manager)

