"""
📥 STEP 1: Ingest Local Documents
Đọc và parse tất cả file PDF/DOCX/TXT trong thư mục ./documents/
Tự động:
 - Kiểm tra incremental (chỉ xử lý file mới hoặc thay đổi)
 - Trích text và metadata đầy đủ
 - Lưu raw data và metadata riêng từng loại
"""

import os
import logging
from data_layer.data_manager import DataManager
from data_layer.document_loader import EnhancedDocumentProcessor

if __name__ == "__main__":
    # ===== Logging setup =====
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("📥 [1] Ingesting local documents...")

    # ===== Initialize managers =====
    data_manager = DataManager()
    processor = EnhancedDocumentProcessor(data_manager=data_manager)

    # ===== Process all supported documents =====
    raw_docs = processor.process_all_documents(force_reprocess=False)

    print(f"✅ Ingested {len(raw_docs)} documents.")

    if not raw_docs:
        print("⚠️ No new or updated documents found. You may force reprocess if needed.")
    else:
        # ===== Save per document type =====
        for doc in raw_docs:
            source_type = doc.get("source", "other").lower()
            subdir = (
                "pdf" if source_type == "pdf" else
                "docx" if source_type == "docx" else
                "text"
            )

            # Lưu dữ liệu raw
            data_manager.save_raw_data([doc], f"local_{source_type}_documents", subdir=subdir)

        # Lưu metadata tổng hợp
        data_manager.save_document_metadata(raw_docs, "local_documents")

        print("📂 Raw data saved to ./data/raw/{pdf,docx,text}/")
        print("🧾 Metadata saved to ./data/metadata/")

    print("✅ Done.")
