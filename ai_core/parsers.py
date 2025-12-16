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
        """
        try:
            print(f"🔴 PARSER RAW INPUT (first 1000 chars): {text[:1000]}")
            
            # Remove any leading/trailing whitespace
            text = text.strip()
            
            # Phát hiện nếu text đã là JSON hợp lệ (bắt đầu bằng { và kết thúc bằng })
            if text.startswith('{') and text.endswith('}'):
                try:
                    parsed = json.loads(text)
                    print(f"🟢 Parsed as clean JSON directly")
                    parsed = self._validate_arbin_structure(parsed)
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"🟡 Direct JSON parse failed, trying cleanup: {e}")
            
            # ====== Pattern 1: Tìm JSON trong code blocks ======
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # ```json { ... } ```
                r'```\s*(\{.*?\})\s*```',      # ``` { ... } ```
            ]
            
            for i, pattern in enumerate(json_patterns):
                print(f"Trying pattern {i}: {pattern}")
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    json_str = matches[0]
                    print(f"Found JSON in pattern {i}: {json_str[:200]}...")
                    try:
                        parsed = json.loads(json_str)
                        print(f"✅ Successfully parsed JSON from pattern {i}")
                        parsed = self._validate_arbin_structure(parsed)
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error pattern {i}: {e}")
                        print(f"JSON string: {json_str[:500]}")
                        # Try to clean and parse again
                        json_str = self._clean_json_string(json_str)
                        try:
                            parsed = json.loads(json_str)
                            print(f"✅ Successfully parsed after cleanup")
                            parsed = self._validate_arbin_structure(parsed)
                            return parsed
                        except:
                            continue
            
            # ====== Pattern 2: Tìm JSON block đơn giản ======
            # Tìm chuỗi bắt đầu bằng { và kết thúc bằng }, có thể có nested {}
            json_block_pattern = r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})'
            matches = re.findall(json_block_pattern, text, re.DOTALL)
            
            for match in matches:
                print(f"Found potential JSON block: {match[:200]}...")
                try:
                    # Clean the string
                    clean_match = self._clean_json_string(match)
                    parsed = json.loads(clean_match)
                    print(f"✅ Parsed from JSON block")
                    parsed = self._validate_arbin_structure(parsed)
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"JSON block parse failed: {e}")
                    continue
            
            # ====== Fallback: Manual extraction ======
            print("⚠️ All JSON parsing methods failed, trying manual extraction")
            extracted_data = self._extract_key_value_pairs(text)
            if extracted_data:
                print(f"🟡 Extracted data manually: {extracted_data}")
                return extracted_data
            
            print("❌ Could not parse any JSON from LLM output")
            return {}
            
        except Exception as e:
            print(f"❌ Parser exception: {e}")
            import traceback
            traceback.print_exc()
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
        Kiểm tra và chuẩn hóa cấu trúc dữ liệu
        """
        print(f"🛠️ VALIDATE STRUCTURE input: {data}")
        
        # Đảm bảo intent tồn tại
        if "intent" not in data:
            # Cố gắng tìm intent từ các field khác
            for key in ["intent", "classification", "type", "category"]:
                if key in data:
                    data["intent"] = data[key]
                    break
            else:
                data["intent"] = "unknown"
        
        # Đảm bảo confidence tồn tại và là số
        if "confidence" not in data:
            # Tự tính confidence nếu không có
            data["confidence"] = 0.7  # Default medium confidence
        else:
            try:
                conf = float(data["confidence"])
                data["confidence"] = max(0.0, min(1.0, conf))
            except:
                data["confidence"] = 0.5
        
        # Đảm bảo entities tồn tại và là dict
        if "entities" in data:
            if not isinstance(data["entities"], dict):
                data["entities"] = {}
        else:
            # Nếu không có entities nhưng có các field entity riêng lẻ
            entity_fields = ["product_names", "technical_terms", "specifications"]
            found_entities = {}
            for field in entity_fields:
                if field in data:
                    found_entities[field] = data.pop(field)
            
            if found_entities:
                data["entities"] = found_entities
            else:
                data["entities"] = {}
        
        print(f"🛠️ VALIDATE STRUCTURE output: {data}")
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
