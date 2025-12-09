from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    TextPreprocessor: Tiền xử lý văn bản thông minh
    Tập trung vào:
    1. Làm sạch văn bản (clean_text) - GIỮ thông tin quan trọng
    2. Chia nhỏ văn bản thành các chunk với boundary-aware
    3. Thêm metadata markers cho các loại thông tin đặc biệt
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Khởi tạo preprocessor với cấu hình chunk

        Args:
            chunk_size: Số từ tối đa trong mỗi chunk (mặc định: 1000)
            chunk_overlap: Số từ overlap giữa các chunk liền kề (mặc định: 200)
        
        Raises:
            ValueError: Nếu tham số không hợp lệ
        """
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
        
        logger.info(f"Initialized TextPreprocessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def _compile_patterns(self):
        """Compile tất cả regex patterns một lần để tăng hiệu năng"""
        # HTML cleaning patterns
        self.script_style_pattern = re.compile(
            r'<(script|style).*?>.*?</\1>',
            flags=re.DOTALL | re.IGNORECASE
        )
        self.html_tags_pattern = re.compile(r'<[^>]+>')
        
        # Whitespace normalization
        self.whitespace_pattern = re.compile(r'[ \t]+')
        self.multiple_newlines_pattern = re.compile(r'\n\s*\n\s*\n+')
        
        # Contact info patterns
        self.phone_pattern = re.compile(
            r'(\+\d{1,3})[\s\-\.]*(\d{1,})[\s\-\.]*(\d{1,})[\s\-\.]*(\d{1,})'
        )
        self.email_pattern = re.compile(
            r'([a-zA-Z0-9._%+-]+)\s*@\s*([a-zA-Z0-9.-]+)\s*\.\s*([a-zA-Z]{2,})'
        )
        
        # Metadata markers - ORDERED by priority (specific to general)
        self.metadata_patterns = [
            # Contact types (most specific first)
            (re.compile(r'\b(Sales Department|Sales Team|Bán hàng|Phòng kinh doanh)\b', re.IGNORECASE), 'DEPT_SALES'),
            (re.compile(r'\b(Support Department|Technical Support|Hỗ trợ kỹ thuật)\b', re.IGNORECASE), 'DEPT_SUPPORT'),
            (re.compile(r'\b(Engineering Department|Kỹ thuật)\b', re.IGNORECASE), 'DEPT_ENGINEERING'),
            (re.compile(r'\b(Marketing Department|Marketing|Tiếp thị)\b', re.IGNORECASE), 'DEPT_MARKETING'),
            
            # Special email types
            (re.compile(r'(sales[\.\-]?[a-z]*@|support[\.\-]?[a-z]*@)', re.IGNORECASE), 'CONTACT_TYPE'),
            
            # Contact info
            (re.compile(r'([\w\.\-]+@[\w\.\-]+\.[\w]+)', re.IGNORECASE), 'EMAIL'),
            (re.compile(r'(\+\d[\d\s\-\(\)\.]{7,})'), 'PHONE'),
            
            # People
            (re.compile(r'\b(Mr\.|Ms\.|Mrs\.|Dr\.|Professor|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'), 'PERSON'),
            
            # Countries
            (re.compile(r'\b(USA|U\.S\.A\.|United States)\b', re.IGNORECASE), 'COUNTRY_USA'),
            (re.compile(r'\b(Germany|Deutschland)\b', re.IGNORECASE), 'COUNTRY_GERMANY'),
            (re.compile(r'\b(China|中国|Beijing|北京)\b', re.IGNORECASE), 'COUNTRY_CHINA'),
            (re.compile(r'\b(India|भारत|Pune|मुंबई)\b', re.IGNORECASE), 'COUNTRY_INDIA'),
            (re.compile(r'\b(Vietnam|Việt Nam|Hanoi|Hồ Chí Minh)\b', re.IGNORECASE), 'COUNTRY_VIETNAM'),
            (re.compile(r'\b(Taiwan|臺灣|台北)\b', re.IGNORECASE), 'COUNTRY_TAIWAN'),
            (re.compile(r'\b(Korea|한국|Seoul|서울)\b', re.IGNORECASE), 'COUNTRY_KOREA'),
            
            # Regions
            (re.compile(r'\b(North America|Northern America)\b', re.IGNORECASE), 'REGION_NA'),
            (re.compile(r'\b(South America|Latin America)\b', re.IGNORECASE), 'REGION_SA'),
            (re.compile(r'\b(Europe|European Union|EU)\b', re.IGNORECASE), 'REGION_EU'),
            (re.compile(r'\b(Middle East|中东|الشرق الأوسط)\b', re.IGNORECASE), 'REGION_ME'),
            (re.compile(r'\b(Africa|非洲|أفريقيا)\b', re.IGNORECASE), 'REGION_AFRICA'),
            (re.compile(r'\b(Southeast Asia|SE Asia|东南亚|Đông Nam Á)\b', re.IGNORECASE), 'REGION_SEA'),
            (re.compile(r'\b(Oceania|Australia|Australasia)\b', re.IGNORECASE), 'REGION_OCEANIA'),
            (re.compile(r'\b(Asia Pacific|APAC)\b', re.IGNORECASE), 'REGION_APAC'),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản NHƯNG GIỮ THÔNG TIN QUAN TRỌNG
        
        Args:
            text: Văn bản đầu vào có thể chứa HTML, định dạng kỳ lạ
            
        Returns:
            Văn bản đã được làm sạch với metadata markers
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_length = len(text)
        
        # 1️⃣ XÓA HTML TAGS NHƯNG GIỮ NỘI DUNG
        text = self.script_style_pattern.sub('', text)
        text = self.html_tags_pattern.sub(' ', text)
        
        # 2️⃣ CHUẨN HÓA KHOẢNG TRẮNG NHƯNG GIỮ CẤU TRÚC
        text = self.whitespace_pattern.sub(' ', text)
        text = self.multiple_newlines_pattern.sub('\n\n', text)
        
        # 3️⃣ CHUẨN HÓA THÔNG TIN CONTACT
        text = self.phone_pattern.sub(r'\1 \2 \3 \4', text)
        text = self.email_pattern.sub(r'\1@\2.\3', text)
        
        # 4️⃣ THÊM METADATA MARKERS - SỬ DỤNG SINGLE PASS
        # Tìm tất cả matches và sắp xếp theo priority
        replacements = []
        
        for pattern, marker in self.metadata_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                match_text = match.group()
                
                # Kiểm tra overlap với các replacements đã tìm thấy
                overlap = False
                for (rep_start, rep_end, _) in replacements:
                    if not (end <= rep_start or start >= rep_end):
                        overlap = True
                        break
                
                if not overlap:
                    replacements.append((start, end, f"[{marker}: {match_text}]"))
        
        # Apply replacements từ cuối lên đầu để giữ nguyên index
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        
        # 5️⃣ FINAL CLEANUP
        text = text.strip()
        
        # Log compression ratio
        if original_length > 0:
            compression_ratio = (1 - len(text) / original_length) * 100
            logger.debug(f"Cleaned text: {original_length:,} → {len(text):,} chars ({compression_ratio:.1f}% reduction)")
        
        return text
    
    def _find_chunk_boundary(self, words: List[str], start_idx: int, max_end_idx: int) -> int:
        """
        Tìm điểm kết thúc chunk tốt nhất, không cắt giữa câu hoặc marker
        
        Args:
            words: Danh sách từ
            start_idx: Vị trí bắt đầu
            max_end_idx: Vị trí kết thúc tối đa (theo chunk_size)
            
        Returns:
            Vị trí kết thúc thích hợp
        """
        if max_end_idx >= len(words):
            return len(words)
        
        # Ưu tiên 1: Không cắt giữa metadata marker [...]
        if '[' in words[max_end_idx - 1] and ']' not in words[max_end_idx - 1]:
            # Tìm dấu đóng marker ]
            for i in range(max_end_idx, min(max_end_idx + 10, len(words))):
                if ']' in words[i]:
                    return i + 1
            return max_end_idx
        
        # Ưu tiên 2: Kết thúc ở dấu câu
        sentence_enders = {'.', '!', '?', '。', '！', '？'}
        for i in range(min(10, max_end_idx - start_idx)):
            check_idx = max_end_idx - i - 1
            if check_idx >= start_idx:
                word = words[check_idx]
                if any(word.endswith(ender) for ender in sentence_enders):
                    return check_idx + 1
        
        # Ưu tiên 3: Kết thúc ở dấu phẩy, chấm phẩy
        comma_enders = {',', ';', '，', '；'}
        for i in range(min(5, max_end_idx - start_idx)):
            check_idx = max_end_idx - i - 1
            if check_idx >= start_idx:
                word = words[check_idx]
                if any(word.endswith(ender) for ender in comma_enders):
                    return check_idx + 1
        
        # Default: return max_end_idx
        return max_end_idx
    
    def split_into_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chia văn bản thành các chunk với boundary-aware splitting
        - Không cắt giữa câu hoặc metadata markers
        - Tạo metadata chi tiết cho mỗi chunk

        Args:
            text: Văn bản đã được clean (nên clean trước)
            metadata: Metadata gốc của document

        Returns:
            List các dict chunk, mỗi chunk có text + metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Split theo từ
        words = text.split()
        if not words:
            return []
        
        chunks = []
        doc_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(words):
            # Xác định end index cơ bản
            basic_end_idx = min(start_idx + self.chunk_size, len(words))
            
            # Tìm boundary tốt hơn
            end_idx = self._find_chunk_boundary(words, start_idx, basic_end_idx)
            
            # Đảm bảo có overlap tối thiểu
            if chunk_num > 0 and end_idx - start_idx < self.chunk_size // 4:
                end_idx = min(start_idx + self.chunk_size // 2, len(words))
            
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Tạo metadata chi tiết cho chunk
            chunk_metadata = {
                **metadata,
                'chunk_id': f"{doc_hash}_{chunk_num}",
                'chunk_index': chunk_num,
                'chunk_start_word': start_idx,
                'chunk_end_word': end_idx - 1,
                'chunk_size_words': len(chunk_words),
                'chunk_size_chars': len(chunk_text),
                'chunk_hash': hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                'total_chunks': -1,  # Sẽ cập nhật sau
                'document_hash': doc_hash,
                'processed_at': datetime.now().isoformat()
            }
            
            chunks.append({
                'id': chunk_metadata['chunk_id'],
                'text': chunk_text,
                'metadata': chunk_metadata,
                'embedding_ready': True
            })
            
            # Di chuyển start index cho chunk tiếp theo
            start_idx = max(start_idx + 1, end_idx - self.chunk_overlap)
            chunk_num += 1
        
        # Cập nhật total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total_chunks
        
        logger.debug(f"Split into {total_chunks} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks
    
    def clean_and_chunk(self, raw_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pipeline chính: Làm sạch và chunk tất cả documents
        - Clean từng document
        - Chunk document đã clean
        - Thu thập metadata

        Args:
            raw_documents: List dict, mỗi dict có 'content' và optional 'metadata'

        Returns:
            List các chunk đã được clean và chunked
        """
        all_chunks = []
        total_docs = len(raw_documents)
        
        if total_docs == 0:
            logger.warning("No documents to process")
            return []
        
        logger.info(f"Starting clean_and_chunk for {total_docs} documents")
        
        for idx, raw_doc in enumerate(raw_documents):
            content = raw_doc.get('content', '')
            doc_metadata = raw_doc.get('metadata', {})
            
            if not content:
                logger.warning(f"Document {idx} has no content, skipping")
                continue
            
            # 1. Bổ sung thông tin cơ bản vào metadata
            enriched_metadata = self._enrich_metadata(raw_doc, doc_metadata, idx)
            
            # 2. Clean document
            try:
                cleaned_content = self.clean_text(content)
            except Exception as e:
                logger.error(f"Error cleaning document {idx}: {e}")
                cleaned_content = content  # Fallback to original
            
            # 3. Chunk document đã clean
            chunks = self.split_into_chunks(cleaned_content, enriched_metadata)
            all_chunks.extend(chunks)
            
            # 4. Log tiến trình
            if (idx + 1) % max(1, total_docs // 10) == 0 or (idx + 1) == total_docs:
                logger.info(f"Processed {idx + 1}/{total_docs} documents, "
                           f"created {len(all_chunks)} chunks so far")
        
        logger.info(f"✅ Completed: {len(all_chunks)} chunks from {total_docs} documents")
        
        # Log thống kê chi tiết
        if all_chunks:
            self._log_statistics(all_chunks)
        
        return all_chunks
    
    def _enrich_metadata(self, raw_doc: Dict[str, Any], metadata: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Bổ sung thông tin metadata từ raw_doc"""
        enriched = metadata.copy()
        
        # Source information
        if 'source' not in enriched:
            enriched['source'] = raw_doc.get('source', 'unknown')
        
        # URL information
        if 'url' not in enriched:
            url = raw_doc.get('url', '')
            if url:
                enriched['url'] = url
        
        # Title information
        if 'title' not in enriched:
            title = raw_doc.get('title') or raw_doc.get('file_name') or f'Document_{idx}'
            enriched['title'] = title
        
        # File information
        if 'file_name' not in enriched:
            file_name = raw_doc.get('file_name', '')
            if file_name:
                enriched['file_name'] = file_name
        
        # Original document index
        enriched['document_index'] = idx
        
        return enriched
    
    def _log_statistics(self, chunks: List[Dict[str, Any]]):
        """Log thống kê chi tiết về chunks"""
        if not chunks:
            return
        
        total_chunks = len(chunks)
        total_words = sum(len(chunk['text'].split()) for chunk in chunks)
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_words = total_words / total_chunks
        avg_chars = total_chars / total_chunks
        
        # Find min/max chunk sizes
        word_counts = [len(chunk['text'].split()) for chunk in chunks]
        char_counts = [len(chunk['text']) for chunk in chunks]
        
        logger.info("📊 Chunk Statistics:")
        logger.info(f"   Total chunks: {total_chunks:,}")
        logger.info(f"   Total words: {total_words:,}")
        logger.info(f"   Total characters: {total_chars:,}")
        logger.info(f"   Average words per chunk: {avg_words:.1f}")
        logger.info(f"   Average chars per chunk: {avg_chars:.1f}")
        logger.info(f"   Min/Max words: {min(word_counts)} / {max(word_counts)}")
        logger.info(f"   Min/Max chars: {min(char_counts)} / {max(char_counts)}")
        logger.info(f"   Config: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Alias để tương thích với code cũ
        
        Args:
            documents: List các document raw
            
        Returns:
            List các chunk đã được xử lý
        """
        return self.clean_and_chunk(documents)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Example documents
    sample_documents = [
        {
            'content': """
            <html>
            <head><title>Company Contact</title></head>
            <body>
            <h1>ABC Corporation</h1>
            <p>For sales inquiries, please contact our Sales Department at sales@abccorp.com</p>
            <p>Phone: +1-800-123-4567</p>
            <p>Support email: support@abccorp.com</p>
            <p>Our offices are in USA, Germany, and Vietnam.</p>
            <script>console.log('test');</script>
            </body>
            </html>
            """,
            'metadata': {
                'source': 'website',
                'url': 'https://abccorp.com/contact'
            }
        },
        {
            'content': """
            Product Specifications Document
            Technical support is available 24/7.
            Contact Dr. John Smith for engineering questions.
            Regional offices in Southeast Asia and Europe.
            Email: info@company.com
            Phone: +84 28 3823 4567
            """,
            'file_name': 'product_specs.txt'
        }
    ]
    
    # Process documents
    chunks = preprocessor.clean_and_chunk(sample_documents)
    
    # Display results
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['id']}")
        print(f"Text: {chunk['text'][:200]}...")
        print(f"Metadata keys: {list(chunk['metadata'].keys())}")