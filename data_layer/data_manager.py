import json
import pickle
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import jsonlines
import time
import csv

logger = logging.getLogger(__name__)

class DataManager:
    """
    Lớp DataManager — trung tâm quản lý dữ liệu của toàn bộ pipeline.
    
    Chức năng chính:
    - Lưu, tải và quản lý dữ liệu ở các giai đoạn khác nhau (raw, processed, embeddings…)
    - Ghi log hoạt động
    - Backup dữ liệu crawler
    - Xuất dữ liệu kiểm tra (inspection)
    - Thống kê và dọn dẹp dữ liệu cũ
    """

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        # Khởi tạo cấu trúc thư mục cơ bản
        self._setup_directories()
        # Khởi tạo file log
        self.log_file = self._init_log_file()

    # ====================================================
    # Setup & Logging
    # ====================================================
    def _setup_directories(self):
        """Tạo sẵn các thư mục chính để đảm bảo không bị lỗi khi ghi file"""
        directories = [
            "raw", "processed", "processed/pdf", "processed/web", "embeddings", "metadata",
            "logs", "inspection", "downloads", "backup"
        ]
        for dir_name in directories:
            path = self.base_dir / dir_name
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataManager initialized at {self.base_dir}")

    def _init_log_file(self):
        """Tạo file log riêng theo ngày để theo dõi hoạt động hệ thống"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"datamanager_{datetime.now().strftime('%Y%m%d')}.log"

        # Ghi log ra file
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Chỉ thêm handler nếu logger chưa có
        if not logger.handlers:
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return log_file

    # ====================================================
    # RAW DATA (dữ liệu gốc)
    # ====================================================
    def save_raw_data(self, data: List[Dict], source: str, subdir: str = None) -> str:
        """
        Lưu dữ liệu thô (ví dụ: kết quả crawl từ web, extract từ PDF, v.v.)
        - Dữ liệu lưu dưới dạng JSON có timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_raw_{timestamp}.json"
        filepath = self.base_dir / "raw" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} raw documents → {filepath}")
        return str(filepath)

    def load_raw_data(self, source: str) -> Optional[List[Dict]]:
        """Tải dữ liệu thô mới nhất của một nguồn (theo timestamp mới nhất)"""
        raw_dir = self.base_dir / "raw"
        files = sorted(raw_dir.glob(f"{source}_raw_*.json"), key=os.path.getmtime, reverse=True)
        if not files:
            logger.warning(f"No previous raw data found for {source}")
            return None
        latest = files[0]
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from {latest}")
        return data

    # ====================================================
    # PROCESSED / CHUNKED DATA
    # ====================================================
    def save_processed_data(self, chunks: List[Dict], processor_type: str, subdir: str = None) -> str:
        """
        Lưu dữ liệu đã được xử lý (sau bước làm sạch hoặc chunking)
        - Lưu cả JSON và JSONL để tiện đọc/ghi và huấn luyện
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.base_dir / "processed" / subdir if subdir else self.base_dir / "processed"
        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{processor_type}_full_documents.json"
        json_path = folder / filename
        jsonl_path = folder / f"{processor_type}_full_documents.jsonl"

        # Ghi dạng JSON (đẹp, dễ đọc)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # Ghi dạng JSONL (từng dòng 1 object)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Saved {len(chunks)} processed chunks to {json_path}")
        return str(json_path)

    # ====================================================
    # EMBEDDINGS
    # ====================================================
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], name: str):
        """
        Lưu embeddings (numpy array) và metadata tương ứng
        - Lưu theo timestamp để dễ versioning
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emb_path = self.base_dir / "embeddings" / f"{name}_embeddings_{timestamp}.npz"
        meta_path = self.base_dir / "embeddings" / f"{name}_metadata_{timestamp}.json"

        # Lưu embeddings nén (npz)
        np.savez_compressed(emb_path, embeddings=embeddings)

        # Lưu metadata
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved embeddings ({embeddings.shape}) and metadata ({len(metadata)})")
        return {'embeddings_path': str(emb_path), 'metadata_path': str(meta_path)}

    def load_embeddings(self, name: str) -> tuple:
        """Tải embeddings và metadata mới nhất của một tập dữ liệu"""
        emb_dir = self.base_dir / "embeddings"
        emb_files = sorted(emb_dir.glob(f"{name}_embeddings_*.npz"), key=os.path.getmtime, reverse=True)
        meta_files = sorted(emb_dir.glob(f"{name}_metadata_*.json"), key=os.path.getmtime, reverse=True)
        if not emb_files or not meta_files:
            logger.warning(f"No embeddings found for {name}")
            return None, None
        emb = np.load(emb_files[0])["embeddings"]
        with open(meta_files[0], 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return emb, meta

    # ====================================================
    # PDF & FILE METADATA
    # ====================================================
    def save_file_metadata(self, file_info: Dict[str, Any]) -> str:
        """
        Lưu metadata của các file đã tải (PDF, datasheet, brochure, v.v.)
        Giúp dễ truy xuất nguồn gốc tài liệu.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.base_dir / "downloads" / f"file_metadata_{timestamp}.json"
        existing = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        existing.append(file_info)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved file metadata: {file_info.get('url', '')}")
        return str(filepath)

    # ====================================================
    # QUEUE BACKUP (hàng đợi crawl)
    # ====================================================
    def save_crawl_queue(self, queue_data: List[Any]):
        """Lưu lại hàng đợi crawler để phục hồi nếu hệ thống bị gián đoạn"""
        filepath = self.base_dir / "backup" / "crawl_queue.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(queue_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved crawl queue backup ({len(queue_data)} URLs)")
        return str(filepath)

    def load_crawl_queue(self) -> Optional[List[Any]]:
        """Khôi phục hàng đợi crawler từ file backup"""
        filepath = self.base_dir / "backup" / "crawl_queue.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                queue = json.load(f)
            logger.info(f"Restored crawl queue ({len(queue)} URLs)")
            return queue
        return None

    # ====================================================
    # METADATA SUMMARY
    # ====================================================
    def save_document_metadata(self, documents: List[Dict], source: str):
        """
        Lưu metadata tổng quan cho toàn bộ tập tài liệu đã xử lý.
        Bao gồm: số lượng documents, tổng số chunks, timestamp, ...
        """
        metadata = {
            'source': source,
            'total_documents': len(documents),
            'total_chunks': sum(len(doc.get('chunks', [])) for doc in documents),
            'timestamp': datetime.now().isoformat()
        }
        filepath = self.base_dir / "metadata" / f"{source}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved document metadata: {filepath}")
        return metadata

    # ====================================================
    # EXPORT / INSPECTION
    # ====================================================
    def export_for_inspection(self, data: Any, filename: str, format: str = 'json'):
        """
        Xuất dữ liệu ra thư mục 'inspection' để kiểm tra thủ công.
        Hỗ trợ nhiều định dạng: json, jsonl, txt, csv.
        """
        export_dir = self.base_dir / "inspection"
        export_dir.mkdir(exist_ok=True)
        filepath = export_dir / f"{filename}.{format}"

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == 'jsonl':
            with jsonlines.open(filepath, 'w') as writer:
                if isinstance(data, list):
                    writer.write_all(data)
        elif format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, list):
                    f.write('\n'.join(str(x) for x in data))
                else:
                    f.write(str(data))
        elif format == 'csv' and isinstance(data, list) and data and isinstance(data[0], dict):
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

        logger.info(f"Exported data for inspection → {filepath}")
        return str(filepath)

    # ====================================================
    # STATS & MAINTENANCE
    # ====================================================
    def get_data_stats(self) -> Dict[str, Any]:
        """Thống kê tổng số file trong từng thư mục dữ liệu"""
        stats = {}
        for folder in ['raw', 'processed', 'embeddings', 'metadata', 'downloads']:
            path = self.base_dir / folder
            files = list(path.glob("*"))
            stats[folder] = len(files)
        return stats

    def cleanup_old_data(self, days_to_keep: int = 7, folder: Optional[str] = None):
        """
        Xóa file dữ liệu cũ hơn X ngày.
        Có thể chọn xóa riêng từng thư mục hoặc toàn bộ.
        """
        now = time.time()
        target_dirs = [folder] if folder else ['raw', 'processed', 'embeddings', 'metadata']
        for d in target_dirs:
            dir_path = self.base_dir / d
            for f in dir_path.glob("*"):
                if now - os.path.getmtime(f) > days_to_keep * 86400:
                    os.remove(f)
                    logger.info(f"Removed old file: {f}")

    # ====================================================
    # Utility helpers
    # ====================================================
    def list_files(self, subdir: str) -> List[str]:
        """Liệt kê tất cả file trong thư mục con cụ thể"""
        path = self.base_dir / subdir
        return [str(p) for p in path.glob("*") if p.is_file()]

    def get_latest_file(self, subdir: str, pattern: str = "*.json") -> Optional[str]:
        """Lấy file mới nhất trong thư mục theo pattern (ví dụ: *.json)"""
        path = self.base_dir / subdir
        files = sorted(path.glob(pattern), key=os.path.getmtime, reverse=True)
        return str(files[0]) if files else None
