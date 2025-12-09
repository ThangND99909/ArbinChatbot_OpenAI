"""
🧹 STEP 2: Làm sạch và chia nhỏ (chunk) dữ liệu đã thu thập
✅ Tự động xử lý cả hai nguồn:
    - PDF / DOCX (enhanced_document_processor_raw_*.json)
    - Web (web_raw_*.json)
Dữ liệu sau khi chunk được lưu vào:
    - data/processed/pdf/
    - data/processed/web/
"""

import os
import json
import glob
import logging
from data_layer.data_manager import DataManager
from data_layer.preprocessor import TextPreprocessor

if __name__ == "__main__":
    # ===== Logging setup =====
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("🧹 [2] Cleaning and chunking all raw data sources...")

    # ===== Khởi tạo DataManager và Preprocessor =====
    data_manager = DataManager()
    preprocessor = TextPreprocessor(chunk_size=800, chunk_overlap=150)

    # ===== Quét thư mục data/raw/ để tìm file raw =====
    raw_dir = "./data/raw"
    raw_files = sorted(
        [os.path.basename(f) 
         for f in glob.glob(os.path.join(raw_dir, "*.json")) 
         if "raw" in f]
    )

    if not raw_files:
        raise FileNotFoundError("❌ No raw files found in ./data/raw/. Please run step 1 first.")

    total_chunks = 0
    summary = []

    print(f"📂 Found {len(raw_files)} raw files to process.\n")

    # ===== Xử lý từng file raw =====
    for raw_file in raw_files:
        raw_path = os.path.join(raw_dir, raw_file)

        # Xác định loại dữ liệu (pdf/documents hoặc web)
        if "enhanced_document_processor_raw" in raw_file.lower():
            subdir = "pdf"
        elif "web" in raw_file.lower():
            subdir = "web"
        else:
            subdir = "other"  # Nếu muốn, có thể skip những file khác

        print(f"📄 Processing raw file: {raw_file} → subdir: {subdir}")

        # Đọc JSON
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_docs = json.load(f)
        except Exception as e:
            logging.error(f"❌ Failed to load {raw_file}: {e}")
            continue

        # Bỏ qua nếu file rỗng hoặc không hợp lệ
        if not isinstance(raw_docs, list) or not raw_docs:
            logging.warning(f"⚠️ Skipped {raw_file}: empty or invalid structure.")
            continue

        # Làm sạch và chia nhỏ
        try:
            chunks = preprocessor.clean_and_chunk(raw_docs)
        except Exception as e:
            logging.error(f"❌ Error cleaning/chunking {raw_file}: {e}")
            continue

        total_chunks += len(chunks)

        # Lưu lại
        try:
            processor_type = "enhanced" if subdir == "pdf" else "web"
            data_manager.save_processed_data(chunks, processor_type=processor_type, subdir=subdir)
            summary.append({"source": subdir, "file": raw_file, "chunks": len(chunks)})
            print(f"✅ Processed {len(chunks)} chunks for {subdir} ({raw_file})\n")
        except Exception as e:
            logging.error(f"❌ Failed to save processed data for {raw_file}: {e}")

    # ===== Tổng kết =====
    print("\n📊 Summary:")
    if summary:
        for s in summary:
            print(f" - {s['file']} → {s['chunks']} chunks ({s['source']})")
        print(f"\n🎯 Done! Total chunks created: {total_chunks}")
        print("👉 Check output folders:")
        print("   - data/processed/pdf/")
        print("   - data/processed/web/")
    else:
        print("⚠️ No valid raw files were processed.")
