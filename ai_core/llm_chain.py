import logging
from langchain.chains import LLMChain
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from ai_core.prompts import QA_PROMPT_TEMPLATE
from typing import Dict, Union, Any, Optional, List
from openai import OpenAI  # OpenAI v1.0.0+
import os
from dotenv import load_dotenv

# ====== TẢI BIẾN MÔI TRƯỜNG (.env) ======
load_dotenv()
logger = logging.getLogger(__name__)

# ====== CẤU HÌNH OPENAI API ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print(f"✅ OpenAI API key found, length: {len(OPENAI_API_KEY)}")
else:
    logger.warning("OpenAI API key not found in .env")
    print("⚠️ WARNING: OPENAI_API_KEY not found in environment variables")

# =========================================================
# =============== LỚP WRAPPER CHO OPENAI LLM ===============
# =========================================================
class OpenAILLM(Runnable):
    """
    Wrapper cho OpenAI (v1.0.0+) để sử dụng trong LangChain
    Kế thừa từ Runnable để tương thích với LLMChain
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Khởi tạo OpenAI client với API key từ environment"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                logger.warning("OpenAI API key not properly configured")
                print("⚠️ WARNING: Please set OPENAI_API_KEY in .env file")
                api_key = "placeholder"  # For initialization without crashing
            
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI client initialized with model: {self.model}")
            print(f"✅ OpenAILLM initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            print(f"❌ OpenAILLM initialization error: {e}")
            self.client = None

    def invoke(self, inputs: Union[str, Dict], config=None, **kwargs) -> str:
        """
        Phương thức chính để gọi OpenAI model.
        """
        
        if "stop" in kwargs:
            kwargs.pop("stop")

        # Xử lý đầu vào
        if isinstance(inputs, dict):
            if 'question' in inputs and 'language' in inputs:
                prompt = f"Question: {inputs['question']}\nLanguage: {inputs['language']}"
            elif 'context' in inputs and 'question' in inputs:
                prompt = QA_PROMPT_TEMPLATE.format(
                    context=inputs['context'],
                    question=inputs['question']
                )
            else:
                prompt = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        else:
            prompt = str(inputs)

        # Kiểm tra client
        if not self.client:
            self._initialize_client()
            if not self.client:
                error_msg = "OpenAI client not available. Please check API key configuration."
                logger.error(error_msg)
                # RAISE EXCEPTION, không trả về string
                raise Exception(error_msg)

        # ================= GỬI REQUEST ĐẾN OPENAI API =================
        try:
            # ====== TỰ ĐỘNG GIỚI HẠN ĐỘ DÀI CÂU TRẢ LỜI ======
            # Ưu tiên ngắn gọn hơn tùy vào loại tác vụ
            max_output_tokens = 500  # mặc định
            if isinstance(inputs, dict):
                if "intent" in inputs.get("task", "").lower():
                    max_output_tokens = 150
                elif "entity" in inputs.get("task", "").lower():
                    max_output_tokens = 200
                elif "comparison" in inputs.get("task", "").lower():
                    max_output_tokens = 700
                elif "qa" in inputs.get("task", "").lower():
                    max_output_tokens = 500
                elif "support" in inputs.get("task", "").lower():
                    max_output_tokens = 400
                else:
                    # Nếu có context dài, giảm bớt để tiết kiệm token
                    context_len = len(inputs.get("context", "")) if "context" in inputs else 0
                    max_output_tokens = 300 if context_len < 2000 else 200

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_output_tokens=max_output_tokens
            )
            
            # ================= XỬ LÝ KẾT QUẢ TRẢ VỀ =================
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                logger.debug(f"OpenAI response received, length: {len(result)}")
            else:
                error_msg = "OpenAI returned empty response"
                logger.warning(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"OpenAI API error: {error_msg}")
            print(f"❌ OpenAI API error: {error_msg}")
            
            # QUAN TRỌNG: RAISE EXCEPTION, không trả về string
            # NLUProcessor cần bắt exception này
            raise Exception(f"OpenAI API error: {error_msg}")

        logger.debug(f"OpenAI response extracted: {result[:200]}...")
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
    
    def __repr__(self):
        return f"OpenAILLM(model='{self.model}', temperature={self.temperature})"


# =========================================================
# ===== LỚP SIMPLELLMMANAGER — FALLBACK KHI OPENAI LỖI ====
# =========================================================
class SimpleLLMManager:
    """
    Fallback LLM đơn giản nếu OpenAI không hoạt động
    Dùng để đảm bảo hệ thống không bị crash
    """

    def __init__(self, model: str = "simple-fallback", temperature: float = 0.1):
        logger.warning(f"Using simple fallback LLM: {model}")
        print(f"⚠️ Using simple fallback LLM: {model}")
        self.model = model
        self.temperature = temperature

    @property
    def llm(self):
        """Trả về self để tương thích với interface của LangChain"""
        return self

    def invoke(self, inputs: Union[str, Dict], **kwargs) -> str:
        """Implement invoke method for compatibility with LangChain"""
        if isinstance(inputs, dict):
            if 'question' in inputs and 'language' in inputs:
                question = inputs['question']
                language = inputs['language']
                return f"Fallback response for question in {language}: {question}"
            elif 'context' in inputs and 'question' in inputs:
                context = inputs['context'][:200] if inputs['context'] else "No context"
                question = inputs['question']
                return f"Based on context: {context}...\n\nQuestion: {question}\n\n(Note: Using fallback LLM)"
            else:
                return f"Fallback response for dict input: {str(inputs)[:100]}..."
        else:
            return f"Fallback LLM response: {str(inputs)[:100]}..."

    def generate_response(self, question: str, context: str = "") -> str:
        """
        Sinh phản hồi đơn giản (mô phỏng LLM thật)
        """
        if context:
            return f"Dựa trên thông tin có sẵn: {context[:200]}...\n\nCâu hỏi: {question}\n\n(Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình OpenAI API key trong file .env để có câu trả lời chính xác hơn)"
        else:
            return f"Tôi nhận được câu hỏi: '{question}'. (Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình OpenAI API key trong file .env để có câu trả lời chính xác hơn)"

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
    Lớp trung tâm quản lý LLM (OpenAI hoặc fallback)
    Tương thích với cấu trúc LLMChain của LangChain
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1, use_openai: bool = True):
        # Nếu có OpenAI API thì dùng OpenAILLM, ngược lại dùng fallback
        if use_openai and OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            try:
                self.llm = OpenAILLM(model=model, temperature=temperature)
                logger.info(f"✅ Using OpenAI LLM: {model}")
                print(f"✅ LLMManager initialized with OpenAI: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                print(f"❌ OpenAI initialization failed, using fallback: {e}")
                self.llm = SimpleLLMManager(model=model, temperature=temperature)
                logger.warning("Falling back to SimpleLLMManager")
        else:
            self.llm = SimpleLLMManager(model=model, temperature=temperature)
            logger.warning("Using fallback SimpleLLMManager")
            print("⚠️ LLMManager using fallback LLM")

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

        try:
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=qa_prompt,
                verbose=False
            )
            logger.info("✅ Initialized LLM chains")
        except Exception as e:
            logger.warning(f"Failed to create LLMChain: {e}")
            print(f"⚠️ LLMChain creation warning: {e}")
            # Fallback chain
            class SimpleQAClient:
                def invoke(self, inputs):
                    context = inputs.get("context", "")
                    question = inputs.get("question", "")
                    return {
                        'text': f"Simple QA response:\nContext: {context[:100]}...\nQuestion: {question}"
                    }
            self.qa_chain = SimpleQAClient()

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
                # Nếu không có context, gọi invoke trực tiếp
                return str(self.llm.invoke(question)).strip()

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            print(f"❌ Error in generate_response: {e}")
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

        try:
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=False
            )
            self.chains[name] = chain
            return chain
        except Exception as e:
            logger.error(f"Failed to create chain '{name}': {e}")
            # Return mock chain
            class MockChain:
                def invoke(self, inputs):
                    return {'text': f"Mock chain '{name}' response"}
            return MockChain()

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
def get_llm_manager(use_openai: bool = True):
    """
    Hàm tiện ích để khởi tạo LLMManager
    Dùng để dễ dàng thay đổi model hoặc fallback
    """
    return LLMManager(use_openai=use_openai)


