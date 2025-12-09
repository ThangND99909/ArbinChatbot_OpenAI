"""
🧠 STEP 3: Tạo embeddings và lưu vào ChromaDB
Tự động nhận cả hai loại dữ liệu:
    - data/processed/pdf/
    - data/processed/web/
"""

import os
import json
import glob
import logging
from data_layer.vector_store import EnhancedVectorStore
from data_layer.data_manager import DataManager

if __name__ == "__main__":
    # ===== Logging setup =====
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("🧠 [3] Embedding and storing to ChromaDB...")

    # ===== Khởi tạo các đối tượng cần thiết =====
    data_manager = DataManager()
    vector_store = EnhancedVectorStore(
        persist_directory="./chroma_db",
        collection_name="arbin_documents"
    )

    # ===== Tìm các file processed JSON =====
    processed_dirs = ["./data/processed/pdf", "./data/processed/web"]
    processed_files = []

    for d in processed_dirs:
        if os.path.exists(d):
            processed_files.extend(glob.glob(os.path.join(d, "*.json")))

    if not processed_files:
        raise FileNotFoundError("❌ No processed files found in data/processed/. Please run step 3 first.")

    print(f"📂 Found {len(processed_files)} processed files to embed.\n")

    total_chunks = 0
    total_new = total_updated = total_duplicates = 0

    # ===== Xử lý từng file =====
    for processed_path in processed_files:
        file_name = os.path.basename(processed_path)
        subdir = "pdf" if "pdf" in processed_path.lower() else "web" if "web" in processed_path.lower() else "other"

        print(f"📄 Embedding {file_name} → source: {subdir}")

        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            logging.error(f"❌ Failed to read {file_name}: {e}")
            continue

        if not isinstance(chunks, list) or not chunks:
            logging.warning(f"⚠️ Skipped {file_name}: invalid or empty data.")
            continue

        # ===== Thêm vào vector store =====
        try:
            result = vector_store.add_document_chunks(chunks)
            total_chunks += len(chunks)
            total_new += result.get("new", 0)
            total_updated += result.get("updated", 0)
            total_duplicates += result.get("duplicates", 0)

            print(f"✅ Embedded {len(chunks)} chunks → {result['status']}")
            print(f"   New: {result.get('new', 0)}, Updated: {result.get('updated', 0)}, Duplicates: {result.get('duplicates', 0)}\n")

        except Exception as e:
            logging.error(f"❌ Error embedding {file_name}: {e}")

    # ===== Xuất thống kê tổng =====
    print("📊 Embedding Summary:")
    print(f"   Total chunks processed: {total_chunks}")
    print(f"   New: {total_new}, Updated: {total_updated}, Duplicates skipped: {total_duplicates}")
    print(f"   Total in store: {vector_store.collection.count()}")
    print(f"   Embedding model: {vector_store.embedding_model_name}")

    # ===== Xuất thống kê ra file inspection =====
    stats = {
        "total_chunks": total_chunks,
        "new": total_new,
        "updated": total_updated,
        "duplicates": total_duplicates,
        "collection_count": vector_store.collection.count(),
        "model": vector_store.embedding_model_name,
    }
    data_manager.export_for_inspection(stats, "embedding_stats", "json")

    print("\n✅ Embedding complete. Check ChromaDB folder: ./chroma_db/")
