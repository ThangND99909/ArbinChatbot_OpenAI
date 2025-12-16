"""
🧠 STEP 4: Tạo embeddings và lưu vào ChromaDB với SentenceTransformer (LOCAL)
"""

import os
import json
import glob
import logging
from tqdm import tqdm
from data_layer.vector_store import EnhancedVectorStore
from data_layer.data_manager import DataManager
import sys
import uuid

# THÊM: Fix encoding cho Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ===== MONKEY PATCH: DISABLE DEDUPLICATION =====
print("⚠️ APPLYING MONKEY PATCH TO DISABLE DEDUPLICATION")

import data_layer.vector_store as vs_module
import numpy as np

# Store original for reference
original_add_documents = vs_module.EnhancedVectorStore.add_documents

def patched_add_documents(self, chunks, batch_size=100, update_existing=True):
    """
    Simplified add_documents without duplicate checking
    - Converts numpy arrays to lists properly
    - Adds all documents without deduplication
    """
    try:
        # Prepare all documents
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        print(f"   Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '').strip()
            metadata = chunk.get('metadata', {})
            metadata = self._validate_metadata(metadata)
            
            # Skip empty or very short text
            if not text or len(text) < 20:
                continue
            
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex}"
            
            all_documents.append(text)
            all_metadatas.append(metadata)
            all_ids.append(doc_id)
            
            # Progress update for large files
            if i > 0 and i % 1000 == 0:
                print(f"     Prepared {i}/{len(chunks)} chunks...")
        
        if not all_documents:
            print("   No valid documents to add")
            return {'status': 'no_changes', 'new': 0, 'updated': 0, 'duplicates': 0}
        
        print(f"   Prepared {len(all_documents)} valid documents for embedding")
        
        # Process in smaller batches to avoid memory issues
        total_added = 0
        small_batch_size = min(50, batch_size)  # Smaller batches for local model
        batch_count = (len(all_documents) + small_batch_size - 1) // small_batch_size
        
        print(f"   Processing in {batch_count} batches of {small_batch_size}...")
        
        for batch_num in range(batch_count):
            start_idx = batch_num * small_batch_size
            end_idx = start_idx + small_batch_size
            
            batch_docs = all_documents[start_idx:end_idx]
            batch_metas = all_metadatas[start_idx:end_idx]
            batch_ids = all_ids[start_idx:end_idx]
            
            if not batch_docs:
                continue
            
            print(f"   Batch {batch_num+1}/{batch_count}: {len(batch_docs)} documents")
            
            try:
                # Create embeddings
                embeddings = self.create_embeddings(batch_docs)
                
                # DEBUG: Check embeddings type
                if embeddings is None:
                    print(f"     ⚠️ No embeddings generated, skipping batch")
                    continue
                
                # Convert numpy arrays to lists
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                elif isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                
                # Ensure embeddings is a list of lists
                if embeddings and not isinstance(embeddings[0], list):
                    embeddings = [embeddings] if isinstance(embeddings, list) else [[embeddings]]
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                total_added += len(batch_docs)
                print(f"     ✓ Added {len(batch_docs)} documents")
                
            except Exception as e:
                error_msg = str(e)
                print(f"     ✗ Batch error: {error_msg[:100]}")
                
                # Try one document at a time as fallback
                successful_in_batch = 0
                for j in range(len(batch_docs)):
                    try:
                        # Get single embedding
                        single_embedding = self.create_embeddings([batch_docs[j]])
                        
                        if single_embedding is None:
                            print(f"       ⚠️ No embedding for document {j+1}, skipping")
                            continue
                        
                        # Convert to list
                        if hasattr(single_embedding, 'tolist'):
                            single_embedding = single_embedding.tolist()
                        elif isinstance(single_embedding, np.ndarray):
                            single_embedding = single_embedding.tolist()
                        
                        # Ensure proper format
                        if isinstance(single_embedding, list) and len(single_embedding) > 0:
                            if not isinstance(single_embedding[0], list):
                                single_embedding = [single_embedding]
                        
                        self.collection.add(
                            embeddings=single_embedding,
                            documents=[batch_docs[j]],
                            metadatas=[batch_metas[j]],
                            ids=[batch_ids[j]]
                        )
                        total_added += 1
                        successful_in_batch += 1
                        
                    except Exception as e2:
                        error_msg2 = str(e2)
                        if "ambiguous" in error_msg2:
                            print(f"       ✗ Skipping document {j+1}: numpy array issue")
                        else:
                            print(f"       ✗ Failed document {j+1}: {error_msg2[:80]}")
                
                if successful_in_batch > 0:
                    print(f"     ➤ Successfully added {successful_in_batch}/{len(batch_docs)} documents from failed batch")
        
        print(f"   Total documents added: {total_added}")
        
        # Update metrics if they exist
        if hasattr(self, 'metrics'):
            self.metrics['total_added'] = total_added
            self.metrics['collection_size'] = self.collection.count()
        
        return {
            'status': 'success' if total_added > 0 else 'partial',
            'new': total_added,
            'updated': 0,
            'duplicates': 0,
            'collection_size': self.collection.count(),
            'total_processed': len(all_documents)
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Critical error in patched_add_documents: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

# Apply the patch
vs_module.EnhancedVectorStore.add_documents = patched_add_documents
print("✅ Deduplication disabled via monkey patch")
print("="*50 + "\n")

# ===== MAIN SCRIPT =====

if __name__ == "__main__":
    # ===== Logging setup =====
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("embedding.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print("🧠 [4] Embedding and storing to ChromaDB with SentenceTransformer (LOCAL)...")
    
    # ===== Khởi tạo các đối tượng cần thiết =====
    data_manager = DataManager()
    
    # Sử dụng SentenceTransformer LOCAL embeddings - KHÔNG CẦN API KEY
    vector_store = EnhancedVectorStore(
        persist_directory="./chroma_db",
        collection_name="arbin_documents",
        embedding_model="intfloat/multilingual-e5-base",  # Model local hỗ trợ tiếng Việt
        embedding_batch_size=256,  # Batch nhỏ hơn cho local model
        max_collection_size=200000,
        enable_backup=True
        # KHÔNG CẦN openai_api_key nữa
    )
    
    # ===== Tìm các file processed JSON =====
    processed_dirs = ["./data/processed/pdf", "./data/processed/web"]
    processed_files = []
    
    for d in processed_dirs:
        if os.path.exists(d):
            processed_files.extend(glob.glob(os.path.join(d, "*.json")))
    
    if not processed_files:
        raise FileNotFoundError("❌ No processed files found. Please run step 3 first.")
    
    print(f"📂 Found {len(processed_files)} processed files to embed.\n")
    
    total_chunks = 0
    total_new = total_updated = total_duplicates = 0
    
    # ===== Xử lý từng file với progress bar =====
    for processed_path in tqdm(processed_files, desc="Processing files"):
        file_name = os.path.basename(processed_path)
        subdir = "pdf" if "pdf" in processed_path.lower() else "web" if "web" in processed_path.lower() else "other"
        
        print(f"\n📄 Embedding {file_name} → source: {subdir}")
        
        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            logging.error(f"❌ Failed to read {file_name}: {e}")
            continue
        
        if not isinstance(chunks, list) or not chunks:
            logging.warning(f"⚠️ Skipped {file_name}: invalid or empty data.")
            continue
        
        total_chunks += len(chunks)
        
        # ===== Thêm vào vector store =====
        try:
            result = vector_store.add_document_chunks(
                chunks,
                batch_size=100,  # Giảm batch size cho local processing
                update_existing=False
            )
            
            total_new += result.get("new", 0)
            total_updated += result.get("updated", 0)
            total_duplicates += result.get("duplicates", 0)
            
            print(f"✅ Embedded {len(chunks)} chunks → {result['status']}")
            print(f"   New: {result.get('new', 0)}, Updated: {result.get('updated', 0)}, Duplicates: {result.get('duplicates', 0)}")
            
            # Show collection size after each file
            current_size = vector_store.collection.count()
            print(f"   Collection size: {current_size} documents")
            
        except Exception as e:
            logging.error(f"❌ Error embedding {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ===== Xuất thống kê tổng =====
    print("\n" + "="*50)
    print("📊 Embedding Summary:")
    print("="*50)
    print(f"   Total chunks processed: {total_chunks}")
    print(f"   New documents: {total_new}")
    print(f"   Updated documents: {total_updated}")
    print(f"   Duplicates skipped: {total_duplicates}")
    
    final_count = vector_store.collection.count()
    print(f"   Total in store: {final_count}")
    
    if final_count < total_new:
        print(f"   ⚠️ Warning: Collection has {final_count} docs but {total_new} were reported as new")
    
    print(f"   Embedding model: {vector_store.embedding_model_name}")
    print(f"   Embedding dimension: {vector_store.embedding_dimension}")
    
    # Lấy thống kê local embedding
    stats = vector_store.get_collection_stats()
    print(f"   Embedding time: {stats.get('embedding_time_seconds', 0):.2f}s")
    print(f"   Embedding speed: {stats.get('embedding_speed', 'N/A')}")
    
    # ===== Xuất thống kê ra file inspection =====
    export_stats = {
        "total_chunks": total_chunks,
        "new": total_new,
        "updated": total_updated,
        "duplicates": total_duplicates,
        "collection_count": final_count,
        "model": vector_store.embedding_model_name,
        "embedding_dimension": vector_store.embedding_dimension,
        "embedding_time_seconds": stats.get('embedding_time_seconds', 0),
        "embedding_speed": stats.get('embedding_speed', 'N/A'),
        "processed_files": len(processed_files),
        "status": "success" if final_count > 100 else "warning_low_docs"
    }
    
    data_manager.export_for_inspection(export_stats, "embedding_stats", "json")
    
    print("\n" + "="*50)
    if final_count > 1000:
        print("✅✅✅ Embedding complete with SUCCESS!")
        print(f"   Vector store now has {final_count} documents")
    elif final_count > 100:
        print("✅ Embedding complete!")
        print(f"   Vector store has {final_count} documents")
    else:
        print("⚠️ Embedding complete but LOW DOCUMENT COUNT!")
        print(f"   Vector store only has {final_count} documents")
        print("   Check deduplication logic and chunk quality")
    
    print(f"✅ Check ChromaDB folder: ./chroma_db/")
    print(f"✅ Stats saved to: ./data/inspection/embedding_stats.json")

    # ===== KIỂM TRA VECTOR STORE SAU KHI EMBEDDING =====
    print("\n" + "="*60)
    print("🔍 KIỂM TRA VECTOR STORE SAU KHI EMBEDDING")
    print("="*60)

    # 1️⃣ Lấy thống kê nhanh
    stats = vector_store.get_collection_stats()
    print(f"📦 Tổng số documents: {stats.get('total_documents', 0)}")
    print(f"📈 Model: {stats.get('embedding_model', 'unknown')}")
    print(f"📏 Dimension: {stats.get('embedding_dimension', 0)}")
    print(f"💾 Lưu tại: {stats.get('persist_directory', './chroma_db')}")
    print(f"🚀 Tốc độ embedding trung bình: {stats.get('embedding_speed', 'N/A')}")

    # 2️⃣ Xem 3 document đầu tiên
    print("\n📄 Xem trước 3 documents đầu tiên trong vector store:\n")
    try:
        results = vector_store.collection.peek(limit=3)
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            text = results['documents'][i][:250].replace("\n", " ") + "..."
            metadata = results['metadatas'][i]
            print(f"--- Document {i+1} ---")
            print(f"ID: {doc_id}")
            print(f"Source: {metadata.get('source', 'unknown')}")
            print(f"Type: {metadata.get('source_type', 'unknown')}")
            print(f"Text: {text}\n")
    except Exception as e:
        print(f"⚠️ Không thể xem trước document: {e}")

    # 3️⃣ Thử search vài truy vấn kiểm chứng
    print("="*60)
    print("🔎 THỬ SEARCH KIỂM TRA TÍNH LIÊN QUAN")
    print("="*60)

    test_queries = [
        "chính sách bảo hành của Arbin",
        "Arbin BT-2000 specifications",
        "lỗi calibration trong phần mềm MITS Pro"
    ]

    for query in test_queries:
        print(f"\n🧩 Truy vấn: {query}")
        try:
            results = vector_store.search_similar(query, k=3)
            if not results:
                print("⚠️ Không tìm thấy kết quả.")
                continue
            for idx, r in enumerate(results):
                text = r['text'][:180].replace("\n", " ")
                score = r['score']
                src = r['metadata'].get('source', 'unknown')
                print(f"  {idx+1}. [{score:.2f}] {text}  (Source: {src})")
        except Exception as e:
            print(f"❌ Lỗi khi search: {e}")

    print("\n✅ Kiểm tra vector store hoàn tất!")
    print("Bạn có thể chạy truy vấn thử bằng chatbot ngay bây giờ 🎯")
    print("="*60)