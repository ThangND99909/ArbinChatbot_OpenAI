import logging
from langchain.chains import LLMChain
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from ai_core.prompts import QA_PROMPT_TEMPLATE
from typing import Dict, Union, Any, Optional, List
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ====== TẢI BIẾN MÔI TRƯỜNG (.env) ======
load_dotenv()
logger = logging.getLogger(__name__)

# ====== CẤU HÌNH GEMINI API ======
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("Gemini API key not found in .env")

# =========================================================
# =============== LỚP WRAPPER CHO GEMINI LLM ===============
# =========================================================
class GeminiLLM(Runnable):
    """
    Wrapper cho Google Gemini để sử dụng trong LangChain
    Kế thừa từ Runnable để tương thích với LLMChain
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        # Nếu có GOOGLE_API_KEY trong môi trường thì cấu hình lại Gemini
        if "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    def invoke(self, inputs: Union[str, Dict], config=None, **kwargs) -> str:
        """
        Phương thức chính để gọi Gemini model.
        Luôn đảm bảo kết quả trả về là `string`.
        """

        if "stop" in kwargs:
            kwargs.pop("stop")  # Loại bỏ tham số không cần thiết (LangChain đôi khi thêm stop token)

        # Xử lý đầu vào: có thể là dict hoặc string
        if isinstance(inputs, dict):
            # Nếu có question + language (dành cho NLU)
            if 'question' in inputs and 'language' in inputs:
                prompt = f"Question: {inputs['question']}\nLanguage: {inputs['language']}"
            # Nếu có context + question (dành cho QA)
            elif 'context' in inputs and 'question' in inputs:
                prompt = QA_PROMPT_TEMPLATE.format(
                    context=inputs['context'],
                    question=inputs['question']
                )
            # Nếu là dict khác, convert toàn bộ thành chuỗi
            else:
                prompt = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        else:
            prompt = str(inputs)

        # Tạo model instance nếu chưa có
        if not hasattr(self, "model_instance"):
            self.model_instance = genai.GenerativeModel(self.model)

        # Gửi prompt đến Gemini API
        response = self.model_instance.generate_content(
            prompt,
            generation_config={"temperature": self.temperature}
        )

        # ================= XỬ LÝ KẾT QUẢ TRẢ VỀ =================
        result = ""
        try:
            # Nếu response có thuộc tính text (cấu trúc chuẩn)
            if hasattr(response, 'text') and response.text:
                result = str(response.text).strip()

            # Nếu không, tìm text trong các candidates
            elif hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                result = str(part.text).strip()
                                break
                    if result:
                        break

            # Nếu vẫn không có, fallback về str(response)
            if not result:
                result = str(response)

        except Exception as e:
            logger.error(f"Error extracting text from Gemini response: {e}")
            result = "I apologize, but I encountered an error processing your request."

        # Đảm bảo luôn trả về string
        if not isinstance(result, str):
            result = str(result)

        logger.debug(f"Gemini response extracted: {result[:200]}...")
        return result

    # Phương thức _call để tương thích với giao diện Runnable của LangChain
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self.invoke(prompt)

    # Xác định kiểu input/output cho Runnable
    @property
    def InputType(self):
        return str

    @property 
    def OutputType(self):
        return str


# =========================================================
# ===== LỚP SIMPLELLMMANAGER — FALLBACK KHI GEMINI LỖI ====
# =========================================================
class SimpleLLMManager:
    """
    Fallback LLM đơn giản nếu Gemini không hoạt động
    Dùng để đảm bảo hệ thống không bị crash
    """

    def __init__(self, model: str = "simple-fallback", temperature: float = 0.1):
        logger.warning(f"Using simple fallback LLM: {model}")
        self.model = model
        self.temperature = temperature

    @property
    def llm(self):
        """Trả về self để tương thích với interface của LangChain"""
        return self

    def generate_response(self, question: str, context: str = "") -> str:
        """
        Sinh phản hồi đơn giản (mô phỏng LLM thật)
        """
        if context:
            return f"Dựa trên thông tin có sẵn: {context[:200]}...\n\nCâu hỏi: {question}\n\n(Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình Gemini API key trong file .env để có câu trả lời chính xác hơn)"
        else:
            return f"Tôi nhận được câu hỏi: '{question}'. (Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình Gemini API key trong file .env để có câu trả lời chính xác hơn)"

    def create_chain(self, name: str, prompt_template: str, input_vars: list):
        """
        Tạo chain giả (mock) để mô phỏng LangChain LLMChain
        """
        class MockChain:
            def invoke(self, inputs):
                question = inputs.get('question', '') if isinstance(inputs, dict) else str(inputs)
                context = inputs.get('context', '') if isinstance(inputs, dict) else ''
                return {'text': f"Mock response for chain: {name}\n\nQuestion: {question}\n\nContext: {context[:100]}..."}

        return MockChain()

    def run_chain(self, name: str, inputs: Dict) -> str:
        """Giả lập việc chạy chain"""
        return f"Mock response from chain '{name}'"

    def predict(self, prompt: str) -> str:
        """Phản hồi đơn giản khi chỉ có prompt"""
        return f"Fallback LLM response: {prompt[:100]}..."


# =========================================================
# =============== LỚP QUẢN LÝ CHÍNH — LLM MANAGER ==========
# =========================================================
class LLMManager:
    """
    Lớp trung tâm quản lý LLM (Gemini hoặc fallback)
    Tương thích với cấu trúc LLMChain của LangChain
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.1, use_gemini: bool = True):
        # Nếu có Gemini API thì dùng GeminiLLM, ngược lại dùng fallback
        if use_gemini and GEMINI_API_KEY:
            self.llm = GeminiLLM(model=model, temperature=temperature)
            logger.info(f"Using Gemini LLM: {model}")
        else:
            self.llm = SimpleLLMManager(model=model, temperature=temperature)
            logger.warning("Using fallback SimpleLLMManager")

        self.chains = {}
        self._init_default_chains()  # Khởi tạo các chain mặc định

    def _init_default_chains(self):
        """
        Khởi tạo chain mặc định: QA (Question Answering)
        """
        qa_prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=qa_prompt,
            verbose=False
        )

        logger.info("Initialized LLM chains")

    def generate_response(self, question: str, context: str = "") -> str:
        """
        Hàm tạo phản hồi chính (được gọi bởi chatbot)
        - Nếu có context → dùng QA chain
        - Nếu không → gọi trực tiếp LLM
        """
        try:
            if context:
                # Gọi chain QA
                result = self.qa_chain.invoke({
                    "context": context,
                    "question": question
                })

                # Đảm bảo trả về string
                if isinstance(result, dict) and 'text' in result:
                    return str(result['text']).strip()
                elif isinstance(result, str):
                    return result.strip()
                else:
                    return str(result).strip()
            else:
                # Nếu không có context, gọi predict trực tiếp
                if hasattr(self.llm, 'predict'):
                    return str(self.llm.predict(question)).strip()
                elif hasattr(self.llm, 'generate_response'):
                    return str(self.llm.generate_response(question)).strip()
                else:
                    return str(f"LLM method not available for question: {question}").strip()

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return str(f"Error: {str(e)[:200]}").strip()

    def create_chain(self, name: str, prompt_template: str, input_vars: list):
        """
        Tạo thêm chain tùy chỉnh mới
        (ví dụ: chain cho sentiment, summarization,...)
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_vars
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )

        self.chains[name] = chain
        return chain

    def run_chain(self, name: str, inputs: Dict) -> str:
        """
        Chạy chain theo tên đã tạo
        """
        chain = self.chains.get(name)
        if not chain:
            raise ValueError(f"Chain '{name}' not found")

        result = chain.invoke(inputs)
        return result['text']


# =========================================================
# =============== FACTORY FUNCTION =========================
# =========================================================
def get_llm_manager(use_gemini: bool = True):
    """
    Hàm tiện ích để khởi tạo LLMManager
    Dùng để dễ dàng thay đổi model hoặc fallback
    """
    return LLMManager(use_gemini=use_gemini)
