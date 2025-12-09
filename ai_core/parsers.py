"""
Parsers for NLU output parsing in Arbin Instruments Chatbot
File này định nghĩa một parser có nhiệm vụ chuyển đổi kết quả từ LLM (văn bản) sang JSON hợp lệ.

Hệ thống có khả năng xử lý lỗi linh hoạt, làm sạch dữ liệu, và chuẩn hóa cấu trúc cho phù hợp với format của Arbin Instruments.

Đồng thời, cung cấp hướng dẫn định dạng (get_format_instructions) để mô hình sinh output đúng chuẩn JSON ngay từ đầu.
"""
import json
import re
import logging
from typing import Dict, Any
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class NLUOutputParser(BaseOutputParser):
    """
    Parser an toàn cho output từ LLM trong hệ thống Arbin Instruments
    
    Chuyển đổi đầu ra text từ LLM sang dict (JSON) với xử lý lỗi mạnh mẽ
    và format instructions tối ưu cho các prompt Arbin
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON output từ LLM với xử lý lỗi mạnh mẽ
        
        Args:
            text: Raw text output từ LLM
            
        Returns:
            Dict[str, Any]: Parsed JSON data hoặc empty dict nếu lỗi
        """
        try:
            # Log phần đầu của output LLM để tiện debug (giới hạn tối đa 500 ký tự)
            log_text = text[:500] + "..." if len(text) > 500 else text
            logger.debug(f"🔹 Raw LLM output: {log_text}")
            
            # ==== Pattern 1: Thử tìm JSON trong markdown code block ====
            # LLM thường trả kết quả trong ```json ... ``` hoặc ``` ... ```
            # hoặc chỉ là { ... } nên ta dò tìm theo các pattern dưới đây
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # ```json { ... } ```
                r'```\s*(.*?)\s*```',      # ``` { ... } ```
                r'\{.*\}',                 # { ... } (bất kỳ)
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    if pattern.startswith('```'):
                        json_str = match.group(1)
                    else:
                        json_str = match.group(0)
                    
                    # Làm sạch chuỗi JSON để tránh lỗi parse
                    json_str = self._clean_json_string(json_str)
                    
                    try:
                        # Thử parse JSON
                        parsed = json.loads(json_str)
                        logger.debug(f"Successfully parsed JSON using pattern")
                        
                        # Chuẩn hóa cấu trúc dữ liệu theo format của Arbin
                        parsed = self._validate_arbin_structure(parsed)
                        
                        return parsed
                    except json.JSONDecodeError as e:
                        # Nếu lỗi JSON thì thử pattern kế tiếp
                        logger.debug(f"JSON decode error: {e}")
                        continue
            
            # ==== Pattern 2: Nếu không tìm thấy JSON trong code block, thử parse toàn bộ text ====
            try:
                text_clean = self._clean_json_string(text)
                parsed = json.loads(text_clean)
                logger.debug("Successfully parsed entire text as JSON")
                
                parsed = self._validate_arbin_structure(parsed)
                return parsed
            except json.JSONDecodeError:
                logger.warning("Không tìm thấy JSON hợp lệ trong output")
                
                # ==== Fallback: Nếu thất bại hoàn toàn, thử trích xuất thủ công các cặp key-value ====
                extracted_data = self._extract_key_value_pairs(text)
                if extracted_data:
                    logger.debug(f"Extracted key-value pairs")
                    return extracted_data
                
                # Không parse được gì thì trả về dict rỗng
                return {}
                
        except Exception as e:
            # Bắt lỗi bất ngờ để tránh crash toàn hệ thống
            logger.error(f"Parse error: {str(e)[:100]}")
            return {}
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Làm sạch JSON string trước khi parse
        - Xóa dấu phẩy thừa ở cuối
        - Đảm bảo key có dấu ngoặc kép
        - Chuyển dấu nháy đơn sang nháy kép
        """
        # Loại bỏ dấu phẩy ở cuối phần tử hoặc object
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Thêm dấu ngoặc kép quanh các key nếu bị thiếu
        json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Chuyển tất cả dấu nháy đơn thành nháy kép để hợp lệ JSON
        json_str = json_str.replace("'", '"')
        
        return json_str
    
    def _validate_arbin_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kiểm tra và chuẩn hóa cấu trúc dữ liệu theo định dạng Arbin
        - Đảm bảo các trường bắt buộc tồn tại
        - Chuyển kiểu dữ liệu về đúng định dạng
        """
        # Nếu có intent mà thiếu confidence → thêm mặc định
        if "intent" in data:
            if "confidence" not in data:
                data["confidence"] = 0.7  # Mặc định độ tin cậy 70%
            if not isinstance(data["confidence"], (int, float)):
                try:
                    data["confidence"] = float(data["confidence"])
                except:
                    data["confidence"] = 0.0
        
        # Nếu có entities nhưng không đúng kiểu dict thì reset về dict rỗng
        if "entities" in data:
            if not isinstance(data["entities"], dict):
                data["entities"] = {}
        
        return data
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """
        Trường hợp LLM không trả JSON hợp lệ, ta fallback bằng cách:
        - Dò các cặp key: value trong văn bản
        - Chuẩn hóa về các field chuẩn của hệ thống Arbin
        """
        result = {}
        
        # Các pattern để tìm key: value theo nhiều kiểu khác nhau
        patterns = [
            r'"([^"]+)"\s*:\s*"([^"]+)"',  # "key": "value"
            r"'([^']+)'\s*:\s*'([^']+)'",  # 'key': 'value'
            r'([a-zA-Z_]+)\s*:\s*"([^"]+)"',  # key: "value"
            r'([a-zA-Z_]+)\s*:\s*([^\s,]+)',  # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key_clean = key.strip().lower()
                value_clean = value.strip()
                
                # Map các key phổ biến về cấu trúc tiêu chuẩn của Arbin
                key_mapping = {
                    "intent": "intent",
                    "confidence": "confidence",
                    "product": "product_names",
                    "model": "product_names",
                    "spec": "specifications",
                    "feature": "features",
                    "application": "applications"
                }
                
                mapped_key = key_mapping.get(key_clean, key_clean)
                
                # Gom giá trị vào dict kết quả
                if mapped_key not in result:
                    # Một số key (product_names, specs, features, apps) nên là list
                    if mapped_key in ["product_names", "specifications", "features", "applications"]:
                        result[mapped_key] = [value_clean]
                    else:
                        result[mapped_key] = value_clean
                elif isinstance(result[mapped_key], list):
                    result[mapped_key].append(value_clean)
        
        return result
    
    def get_format_instructions(self) -> str:
        """
        Trả về hướng dẫn định dạng JSON cho LLM
        → Dùng trong prompt để ép mô hình trả về JSON hợp lệ đúng cấu trúc Arbin
        """
        return """TRẢ LỜI DƯỚI DẠNG JSON HỢP LỆ, theo một trong các format sau:

1. Cho INTENT DETECTION:
{
  "intent": "tên_intent",
  "confidence": số_từ_0_đến_1,
  "alternative_intents": ["intent_2", "intent_3"],
  "explanation": "giải_thích_ngắn"
}

2. Cho ENTITY EXTRACTION:
{
  "entities": {
    "product_names": ["BT-2000", "MITS Pro"],
    "technical_terms": ["voltage", "current"],
    "specifications": ["5V", "10A"],
    "applications": ["battery testing", "R&D"],
    "features": ["high precision", "safety"],
    "issues": ["calibration error", "connection problem"],
    "software_components": ["MITS Pro", "WinDaq"]
  },
  "confidence": 0.8,
  "extraction_notes": "ghi_chú_về_trích_xuất"
}

KHÔNG thêm text bên ngoài JSON. Chỉ trả về JSON."""


# Factory function để tạo parser
def create_nlu_output_parser() -> NLUOutputParser:
    """Factory function để tạo NLUOutputParser instance"""
    return NLUOutputParser()


# Export
__all__ = ['NLUOutputParser', 'create_nlu_output_parser']
