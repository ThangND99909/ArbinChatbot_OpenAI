import logging
from langchain.chains import LLMChain
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from ai_core.prompts import QA_PROMPT_TEMPLATE
from typing import Dict, Union, Any, Optional, List
from openai import OpenAI  # OpenAI v1.0.0+
import os


# ====== TẢI BIẾN MÔI TRƯỜNG (.env) ======

logger = logging.getLogger(__name__)

OPENAI_API_KEY = None  # Sẽ được set sau



# =========================================================
# =============== LỚP WRAPPER CHO OPENAI LLM ===============
# =========================================================
class OpenAILLM(Runnable):
    """
    Wrapper cho OpenAI (v1.0.0+) để sử dụng trong LangChain
    Kế thừa từ Runnable để tương thích với LLMChain
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Khởi tạo OpenAI client với API key từ environment"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
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
                max_tokens=max_output_tokens
            )
            
            # ====== LOG THÔNG TIN SỬ DỤNG TOKEN =====
            if hasattr(response, "usage"):
                usage = response.usage
                print(f"🔢 Tokens used: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
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
class SimpleLLMManager(Runnable):
    """
    Fallback LLM đơn giản nếu OpenAI không hoạt động
    Dùng để đảm bảo hệ thống không bị crash
    Hoàn toàn tương thích với LangChain Runnable interface
    """

    def __init__(self, model: str = "simple-fallback", temperature: float = 0.1):
        logger.warning(f"Using simple fallback LLM: {model}")
        print(f"⚠️ Using simple fallback LLM: {model}")
        self.model = model
        self.temperature = temperature
        self.is_fallback = True  # Flag để nhận biết đang dùng fallback

    # ========== CORE RUNNABLE INTERFACE METHODS ==========
    
    def invoke(self, inputs: Union[str, Dict], config: Optional[Dict] = None, **kwargs) -> str:
        """
        Implement invoke method với đúng signature của Runnable
        LangChain sẽ gọi method này với config parameter
        """
        # Debug logging
        print(f"🔧 SimpleLLMManager.invoke() called")
        print(f"   Input type: {type(inputs)}")
        if isinstance(inputs, dict):
            print(f"   Input keys: {list(inputs.keys())}")
        else:
            print(f"   Input: {str(inputs)[:100]}...")
        
        # Xử lý inputs theo các trường hợp
        if isinstance(inputs, dict):
            # TRƯỜNG HỢP 1: NLU Intent Detection
            if 'question' in inputs and 'language' in inputs:
                question = inputs['question']
                language = inputs['language']
                print(f"   NLU Intent Detection format detected")
                
                # Trả về JSON hợp lệ cho intent detection
                return '{"intent": "unknown", "confidence": 0.5, "alternative_intents": [], "explanation": "Using fallback LLM for intent detection"}'
            
            # TRƯỜNG HỢP 2: NLU Entity Extraction
            elif 'question' in inputs:
                question = inputs['question']
                print(f"   NLU Entity Extraction format detected")
                
                # Trả về JSON hợp lệ cho entity extraction
                return '''{
  "entities": {
    "product_names": [],
    "technical_info": [],
    "applications": [],
    "features": [],
    "issues": [],
    "software": [],
    "locations": []
  },
  "confidence": 0.4,
  "extraction_notes": "Fallback entity extraction - no entities detected"
}'''
            
            # TRƯỜNG HỢP 3: QA Chain (context + question)
            elif 'context' in inputs and 'question' in inputs:
                context = inputs['context'][:200] if inputs['context'] else "No context"
                question = inputs['question']
                print(f"   QA Chain format detected")
                
                return f"""Based on context: {context}...

Question: {question}

Response: I'm currently using a fallback LLM. Please configure your OpenAI API key in the .env file for accurate responses about Arbin Instruments products.

Suggested next steps:
1. Check your .env file has OPENAI_API_KEY
2. Visit www.arbin.com for product information
3. Contact support@arbin.com for technical assistance"""
            
            # TRƯỜNG HỢP 4: Generic dict input
            else:
                return f"Fallback response for dictionary input: {str(inputs)[:100]}..."
        
        else:
            # TRƯỜNG HỢP 5: String input
            input_str = str(inputs)
            return f"""Fallback LLM Response:

You asked: "{input_str[:100]}..."

Note: I'm currently running in fallback mode. To get accurate information about Arbin Instruments products (BT series, MITS Pro, battery testing systems), please:

1. Configure OpenAI API key in .env file
2. Ensure vector store has relevant documents
3. Contact technical support if issues persist

For immediate assistance, email: support@arbin.com"""

    # Phương thức _call để tương thích với giao diện Runnable cũ của LangChain
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        print(f"🔧 SimpleLLMManager._call() called")
        return self.invoke(prompt)

    # Batch invoke để hỗ trợ batch processing
    def batch(self, inputs: List[Union[str, Dict]], config: Optional[Dict] = None, **kwargs) -> List[str]:
        print(f"🔧 SimpleLLMManager.batch() called with {len(inputs)} inputs")
        return [self.invoke(inp, config, **kwargs) for inp in inputs]

    # Stream method (optional, for streaming support)
    def stream(self, input: Union[str, Dict], config: Optional[Dict] = None, **kwargs):
        print(f"🔧 SimpleLLMManager.stream() called")
        yield self.invoke(input, config, **kwargs)

    # Xác định kiểu input/output cho Runnable
    @property
    def InputType(self):
        from typing import Union
        return Union[str, Dict]

    @property 
    def OutputType(self):
        return str
    
    def __repr__(self):
        return f"SimpleLLMManager(model='{self.model}', temperature={self.temperature})"
    
    def __str__(self):
        return f"Fallback LLM Manager (model: {self.model})"
    
    # ========== COMPATIBILITY METHODS ==========
    
    @property
    def llm(self):
        """Trả về self để tương thích với interface của LangChain"""
        return self

    def generate_response(self, question: str, context: str = "") -> str:
        """
        Sinh phản hồi đơn giản (mô phỏng LLM thật)
        Giữ nguyên cho backward compatibility
        """
        if context:
            return f"""Dựa trên thông tin có sẵn: {context[:200]}...

Câu hỏi: {question}

(Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình OpenAI API key trong file .env để có câu trả lời chính xác hơn)"""
        else:
            return f"""Tôi nhận được câu hỏi: '{question}'.

(Lưu ý: Đang sử dụng fallback LLM, vui lòng cấu hình OpenAI API key trong file .env để có câu trả lời chính xác hơn.

Thông tin về Arbin Instruments:
- Website: www.arbin.com
- Hỗ trợ kỹ thuật: support@arbin.com
- Sản phẩm: BT series, MITS Pro, battery testing systems)"""

    def create_chain(self, name: str, prompt_template: str, input_vars: list):
        """
        Tạo chain giả (mock) để mô phỏng LangChain LLMChain
        """
        class MockChain:
            def __init__(self, name):
                self.name = name
            
            def invoke(self, inputs, config=None, **kwargs):
                print(f"🔧 MockChain '{self.name}'.invoke() called")
                
                if isinstance(inputs, dict):
                    question = inputs.get('question', '')
                    context = inputs.get('context', '')
                    return {
                        'text': f"Mock response for chain: {self.name}\n\nQuestion: {question}\n\nContext: {context[:100]}..."
                    }
                else:
                    return {
                        'text': f"Mock response for chain: {self.name}\n\nInput: {str(inputs)[:100]}..."
                    }
            
            def batch(self, inputs_list, config=None, **kwargs):
                return [self.invoke(inp, config, **kwargs) for inp in inputs_list]

        return MockChain(name)

    def run_chain(self, name: str, inputs: Dict) -> str:
        """Giả lập việc chạy chain"""
        print(f"🔧 SimpleLLMManager.run_chain('{name}')")
        return f"Mock response from chain '{name}': {str(inputs)[:100]}..."

    def predict(self, prompt: str) -> str:
        """Phản hồi đơn giản khi chỉ có prompt"""
        print(f"🔧 SimpleLLMManager.predict()")
        return f"Fallback LLM response: {prompt[:100]}..."
    
    # ========== ADDITIONAL HELPER METHODS ==========
    
    def get_model_info(self) -> Dict[str, Any]:
        """Trả về thông tin model"""
        return {
            "model_type": "fallback",
            "model_name": self.model,
            "temperature": self.temperature,
            "is_fallback": True,
            "capabilities": ["text_generation", "intent_detection", "entity_extraction"]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái health của LLM"""
        return {
            "status": "operational",
            "mode": "fallback",
            "message": "Fallback LLM is running. Configure OpenAI API for full functionality.",
            "model": self.model
        }


# =========================================================
# =============== LỚP QUẢN LÝ CHÍNH — LLM MANAGER ==========
# =========================================================
class LLMManager:
    """
    Lớp trung tâm quản lý LLM (OpenAI hoặc fallback)
    Tương thích với cấu trúc LLMChain của LangChain
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, use_openai: bool = True):
        self.model = model
        self.temperature = temperature
        self.use_openai = use_openai
        self.chains = {}

        # ===== Kiểm tra API key và khởi tạo LLM =====
        print(f"\n🔄 Initializing LLMManager...")
        print(f"   Model: {model}")
        print(f"   Use OpenAI: {use_openai}")
        
        # Load .env để check API key
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if use_openai and api_key and len(api_key) > 20 and not api_key.startswith("your_"):
            try:
                print("   Attempting to initialize OpenAI...")
                self.llm = OpenAILLM(model=model, temperature=temperature)
                logger.info(f"✅ Using OpenAI LLM: {model}")
                print(f"✅ LLMManager initialized with OpenAI: {model}")
            except Exception as e:
                print(f"❌ OpenAI init failed: {e}")
                print("🔄 Switching to fallback LLM")
                self.llm = SimpleLLMManager(model=model, temperature=temperature)
        else:
            print("⚠️ OpenAI API key not found or invalid. Using fallback LLM.")
            self.llm = SimpleLLMManager(model=model, temperature=temperature)

        # ===== Khởi tạo các chain mặc định =====
        try:
            self._init_default_chains()
            logger.info("✅ Default LLM chains initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize default chains: {e}")
            print(f"⚠️ Warning: Could not initialize default chains ({e}).")

        # ===== Khởi tạo các chain mặc định =====
        try:
            self._init_default_chains()
            logger.info("✅ Default LLM chains initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize default chains: {e}")
            print(f"⚠️ Warning: Could not initialize default chains ({e}).")

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


