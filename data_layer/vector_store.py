import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Set, Tuple, Optional, Union
import logging
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading
import shutil
import gc
from collections import defaultdict
import math
import re

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """
    EnhancedVectorStore: Quản lý vector store với ChromaDB + embedding model
    - Hỗ trợ deduplication (url, file, content hash)
    - Hỗ trợ thêm document, cập nhật, search tương tự
    - Tương thích với LangChain retriever
    - Thread-safe và production-ready
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "arbin_documents",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 embedding_batch_size: int = 32,
                 max_collection_size: int = 100000,
                 enable_backup: bool = True):
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Configuration
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.max_collection_size = max_collection_size
        self.enable_backup = enable_backup
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.metrics = {
            'total_added': 0,
            'total_updated': 0,
            'total_duplicates': 0,
            'search_count': 0,
            'embedding_time': 0,
            'errors': 0,
            'last_backup': None,
            'collection_size': 0
        }
        
        # Cache for frequent queries
        self.query_cache = {}
        self.max_cache_size = 1000
        
        # Deduplication tracking
        self.processed_urls: Set[str] = set()
        self.processed_files: Set[str] = set()
        self.content_hashes: Dict[str, Tuple[str, datetime]] = {}  # doc_id -> (hash, timestamp)
        self.url_to_id: Dict[str, str] = {}
        self.file_to_id: Dict[str, str] = {}
        self.id_to_source: Dict[str, Dict] = {}  # doc_id -> source info
        
        # Load processed data
        self._load_processed_data()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model with memory optimization
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device='cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        )
        
        # Get or create collection
        with self.lock:
            try:
                self.collection = self.client.get_collection(collection_name)
                self.metrics['collection_size'] = self.collection.count()
                logger.info(f"Loaded existing collection: {collection_name} with {self.metrics['collection_size']} documents")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "created_at": datetime.now().isoformat(),
                        "embedding_model": embedding_model
                    },
                    embedding_function=None  # We handle embeddings manually
                )
                logger.info(f"Created new collection: {collection_name}")
        
        # Start background cleanup thread
        self.cleanup_thread = None
        self.running = True
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background thread for cleanup tasks"""
        def cleanup_worker():
            while self.running:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_data()
                    self._cleanup_cache()
                    if self.enable_backup:
                        self._auto_backup()
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
        
        import time
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_old_data(self, max_age_days: int = 30):
        """Cleanup old data to prevent memory leaks"""
        with self.lock:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            old_hashes = [
                doc_id for doc_id, (_, timestamp) in self.content_hashes.items()
                if timestamp < cutoff_date
            ]
            
            for doc_id in old_hashes:
                if doc_id in self.content_hashes:
                    del self.content_hashes[doc_id]
            
            if old_hashes:
                logger.info(f"Cleaned up {len(old_hashes)} old content hashes")
                self._save_processed_data()
    
    def _cleanup_cache(self):
        """Cleanup query cache"""
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.query_cache.keys())[:len(self.query_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.query_cache[key]
    
    def _load_processed_data(self):
        """Load processed data từ file JSON"""
        try:
            processed_file = self.persist_directory / "processed_data.json"
            if processed_file.exists():
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    
                    # Convert loaded timestamps back to datetime objects
                    self.processed_urls = set(data.get('processed_urls', []))
                    self.processed_files = set(data.get('processed_files', []))
                    
                    # Convert timestamp strings to datetime objects
                    content_hashes = data.get('content_hashes', {})
                    for doc_id, (hash_str, timestamp_str) in content_hashes.items():
                        timestamp = datetime.fromisoformat(timestamp_str)
                        self.content_hashes[doc_id] = (hash_str, timestamp)
                    
                    self.url_to_id = data.get('url_to_id', {})
                    self.file_to_id = data.get('file_to_id', {})
                    self.id_to_source = data.get('id_to_source', {})
                
                logger.info(f"Loaded processed data: {len(self.processed_urls)} URLs, "
                          f"{len(self.processed_files)} files, {len(self.content_hashes)} hashes")
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}")
    
    def _save_processed_data(self):
        """Lưu processed data vào file JSON"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_hashes = {}
            for doc_id, (hash_str, timestamp) in self.content_hashes.items():
                serializable_hashes[doc_id] = (hash_str, timestamp.isoformat())
            
            data = {
                'processed_urls': list(self.processed_urls),
                'processed_files': list(self.processed_files),
                'content_hashes': serializable_hashes,
                'url_to_id': self.url_to_id,
                'file_to_id': self.file_to_id,
                'id_to_source': self.id_to_source,
                'saved_at': datetime.now().isoformat(),
                'metrics': self.metrics
            }
            
            processed_file = self.persist_directory / "processed_data.json"
            
            # Save to temp file first, then rename (atomic write)
            temp_file = self.persist_directory / "processed_data.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            temp_file.replace(processed_file)
            
            logger.debug("Saved processed data")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def get_content_hash(self, content: str) -> str:
        """Tạo hash MD5 cho nội dung (dùng cho deduplication)"""
        # Normalize content (remove extra whitespace, normalize unicode)
        normalized = re.sub(r'\s+', ' ', content.strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def generate_doc_id(self, source_type: str, identifier: str) -> str:
        """Tạo unique doc_id dựa trên source và identifier"""
        # Clean identifier
        clean_identifier = re.sub(r'[^\w\-_.]', '_', identifier)
        timestamp = int(datetime.now().timestamp())
        unique_string = f"{source_type}_{clean_identifier}_{timestamp}"
        unique_hash = hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:12]
        return f"{source_type[:3]}_{unique_hash}"
    
    def _validate_metadata(self, metadata: Dict) -> Dict:
        """Đảm bảo metadata có đầy đủ trường bắt buộc"""
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Ensure required fields
        if 'source_type' not in metadata:
            metadata['source_type'] = 'unknown'
        if 'source' not in metadata:
            metadata['source'] = 'unknown'
        
        # Add timestamp
        metadata['ingested_at'] = datetime.now().isoformat()
        
        # Clean metadata for ChromaDB (only primitive types)
        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            elif value is None:
                cleaned_metadata[key] = ''
            else:
                cleaned_metadata[key] = str(value)
        
        return cleaned_metadata
    
    def is_duplicate(self, source_type: str, identifier: str, content: str) -> Tuple[bool, str]:
        """
        Kiểm tra duplicate với content similarity check
        Returns: (is_duplicate, doc_id_if_exists)
        """
        with self.lock:
            # Check URL/file already processed
            if source_type == "web" and identifier in self.processed_urls:
                return True, self.url_to_id.get(identifier, "")
            elif source_type == "document" and identifier in self.processed_files:
                return True, self.file_to_id.get(identifier, "")
            
            # Check content hash
            content_hash = self.get_content_hash(content)
            for existing_id, (existing_hash, _) in self.content_hashes.items():
                if existing_hash == content_hash:
                    logger.info(f"Duplicate content detected (same hash): {identifier}")
                    return True, existing_id
            
            return False, ""
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Tạo embeddings cho nhiều văn bản với batch processing"""
        import time
        start_time = time.time()
        
        logger.info(f"Creating embeddings for {len(texts)} texts in batches of {self.embedding_batch_size}")
        
        all_embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            
            # Filter out empty or very short texts
            valid_batch = [text for text in batch if text and len(text.strip()) > 10]
            if not valid_batch:
                continue
            
            batch_embeddings = self.embedding_model.encode(
                valid_batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # Important for cosine similarity
                convert_to_numpy=True
            )
            
            # Handle cases where some texts were filtered out
            if len(batch_embeddings) < len(batch):
                # Pad with zeros for filtered texts
                padded_embeddings = np.zeros((len(batch), batch_embeddings.shape[1]))
                valid_idx = 0
                for j, text in enumerate(batch):
                    if text and len(text.strip()) > 10:
                        padded_embeddings[j] = batch_embeddings[valid_idx]
                        valid_idx += 1
                batch_embeddings = padded_embeddings
            
            all_embeddings.append(batch_embeddings)
        
        if not all_embeddings:
            return np.array([])
        
        embeddings = np.vstack(all_embeddings)
        
        elapsed_time = time.time() - start_time
        self.metrics['embedding_time'] += elapsed_time
        
        logger.info(f"Created embeddings: {embeddings.shape} in {elapsed_time:.2f}s")
        return embeddings
    
    def add_documents(self, 
                     chunks: List[Dict], 
                     batch_size: int = 100,
                     update_existing: bool = True) -> Dict[str, Any]:
        """
        Thêm document chunks vào vector store với deduplication và batch processing
        """
        with self.lock:
            try:
                # Check collection size limit
                current_size = self.collection.count()
                if current_size + len(chunks) > self.max_collection_size:
                    logger.warning(f"Collection size limit reached ({self.max_collection_size}). Performing cleanup.")
                    self._enforce_size_limit()
                
                # Prepare batches
                documents = []
                metadatas = []
                ids = []
                
                new_count = 0
                update_count = 0
                duplicate_count = 0
                error_count = 0
                
                for chunk in chunks:
                    try:
                        content = chunk.get('text', '').strip()
                        metadata = chunk.get('metadata', {})
                        
                        if not content or len(content) < 20:
                            logger.warning(f"Skipping chunk with insufficient content")
                            continue
                        
                        # Validate and clean metadata
                        metadata = self._validate_metadata(metadata)
                        source_type = metadata.get('source_type', 'unknown')
                        identifier = ""
                        
                        if source_type == "web":
                            identifier = metadata.get('url', '')
                        elif source_type == "document":
                            identifier = metadata.get('file_path', metadata.get('source', ''))
                        else:
                            identifier = metadata.get('source', '')
                        
                        # Check for duplicates
                        is_dup, existing_id = self.is_duplicate(source_type, identifier, content)
                        
                        if is_dup and existing_id:
                            duplicate_count += 1
                            
                            if update_existing:
                                # Check if content has actually changed
                                old_hash, _ = self.content_hashes.get(existing_id, ('', None))
                                new_hash = self.get_content_hash(content)
                                
                                if old_hash != new_hash:
                                    update_count += 1
                                    metadata['updated'] = True
                                    metadata['update_time'] = datetime.now().isoformat()
                                    metadata['previous_hash'] = old_hash
                                    doc_id = existing_id
                                else:
                                    continue  # Skip identical content
                            else:
                                continue  # Skip duplicates
                        else:
                            # New document
                            new_count += 1
                            doc_id = self.generate_doc_id(source_type, identifier)
                            
                            # Update tracking
                            if source_type == "web" and identifier:
                                self.processed_urls.add(identifier)
                                self.url_to_id[identifier] = doc_id
                            elif source_type == "document" and identifier:
                                self.processed_files.add(identifier)
                                self.file_to_id[identifier] = doc_id
                        
                        # Update content hash with timestamp
                        self.content_hashes[doc_id] = (self.get_content_hash(content), datetime.now())
                        
                        # Store source info
                        self.id_to_source[doc_id] = {
                            'source_type': source_type,
                            'identifier': identifier,
                            'added_at': datetime.now().isoformat()
                        }
                        
                        # Add to batch
                        documents.append(content)
                        metadatas.append(metadata)
                        ids.append(doc_id)
                        
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error processing chunk: {e}")
                        continue
                
                if not documents:
                    logger.info("No documents to add")
                    return {
                        'status': 'no_changes',
                        'new': 0,
                        'updated': 0,
                        'duplicates': duplicate_count,
                        'errors': error_count
                    }
                
                # Save processed data
                self._save_processed_data()
                
                # Process in batches to avoid memory issues
                total_added = 0
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_metadatas = metadatas[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]
                    
                    # Create embeddings for batch
                    batch_embeddings = self.create_embeddings(batch_docs)
                    
                    if len(batch_embeddings) == 0:
                        logger.warning(f"No embeddings created for batch {i//batch_size}")
                        continue
                    
                    try:
                        # Check which IDs already exist
                        existing_ids = set(self.collection.get()['ids'])
                        
                        add_ids = []
                        update_ids = []
                        add_embeddings = []
                        update_embeddings = []
                        add_documents = []
                        update_documents = []
                        add_metadatas = []
                        update_metadatas = []
                        
                        for idx, doc_id in enumerate(batch_ids):
                            if doc_id in existing_ids:
                                update_ids.append(doc_id)
                                update_embeddings.append(batch_embeddings[idx])
                                update_documents.append(batch_docs[idx])
                                update_metadatas.append(batch_metadatas[idx])
                            else:
                                add_ids.append(doc_id)
                                add_embeddings.append(batch_embeddings[idx])
                                add_documents.append(batch_docs[idx])
                                add_metadatas.append(batch_metadatas[idx])
                        
                        # Add new documents
                        if add_ids:
                            self.collection.add(
                                embeddings=add_embeddings,
                                documents=add_documents,
                                metadatas=add_metadatas,
                                ids=add_ids
                            )
                            total_added += len(add_ids)
                        
                        # Update existing documents
                        if update_ids and update_existing:
                            for j, doc_id in enumerate(update_ids):
                                self.collection.update(
                                    embeddings=[update_embeddings[j]],
                                    documents=[update_documents[j]],
                                    metadatas=[update_metadatas[j]],
                                    ids=[doc_id]
                                )
                        
                        logger.info(f"Processed batch {i//batch_size + 1}: "
                                  f"{len(add_ids)} added, {len(update_ids)} updated")
                        
                    except Exception as e:
                        logger.error(f"Error adding batch {i//batch_size}: {e}")
                        # Try individual documents
                        for j, doc_id in enumerate(batch_ids):
                            try:
                                self.collection.upsert(
                                    embeddings=[batch_embeddings[j].tolist()],
                                    documents=[batch_docs[j]],
                                    metadatas=[batch_metadatas[j]],
                                    ids=[doc_id]
                                )
                                total_added += 1
                            except Exception as e2:
                                logger.error(f"Failed to add document {doc_id}: {e2}")
                
                # Update metrics
                self.metrics['total_added'] += new_count
                self.metrics['total_updated'] += update_count
                self.metrics['total_duplicates'] += duplicate_count
                self.metrics['collection_size'] = self.collection.count()
                
                # Clear query cache since collection changed
                self.query_cache.clear()
                
                # Clear embeddings cache to free memory
                self.clear_embeddings_cache()
                
                logger.info(f"Document addition complete: "
                          f"{new_count} new, {update_count} updated, "
                          f"{duplicate_count} duplicates, {error_count} errors")
                
                return {
                    'status': 'success',
                    'total_added': total_added,
                    'new': new_count,
                    'updated': update_count,
                    'duplicates': duplicate_count,
                    'errors': error_count,
                    'collection_size': self.collection.count()
                }
                
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                self.metrics['errors'] += 1
                return {
                    'status': 'error',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
    
    def _enforce_size_limit(self):
        """Enforce collection size limit by removing oldest documents"""
        try:
            # Get all documents with timestamps
            all_docs = self.collection.get(include=['metadatas'])
            
            if not all_docs['ids']:
                return
            
            # Sort by timestamp (oldest first)
            docs_with_times = []
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                timestamp_str = metadata.get('ingested_at', '')
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except:
                    timestamp = datetime.min
                
                docs_with_times.append((doc_id, timestamp))
            
            docs_with_times.sort(key=lambda x: x[1])
            
            # Calculate how many to remove
            excess = len(docs_with_times) - self.max_collection_size
            if excess <= 0:
                return
            
            # Remove oldest documents
            to_remove = docs_with_times[:excess]
            remove_ids = [doc_id for doc_id, _ in to_remove]
            
            logger.info(f"Removing {len(remove_ids)} oldest documents to enforce size limit")
            self.collection.delete(ids=remove_ids)
            
            # Also clean up tracking data
            for doc_id in remove_ids:
                if doc_id in self.content_hashes:
                    del self.content_hashes[doc_id]
                if doc_id in self.id_to_source:
                    del self.id_to_source[doc_id]
            
            self._save_processed_data()
            
        except Exception as e:
            logger.error(f"Error enforcing size limit: {e}")
    
    def search_similar(self, 
                      query: str, 
                      k: int = 5, 
                      score_threshold: float = 0.7,
                      filter_metadata: Dict = None,
                      use_cache: bool = True) -> List[Dict]:
        """Tìm các document tương tự query với caching và filtering"""
        with self.lock:
            try:
                self.metrics['search_count'] += 1
                
                # Check cache
                cache_key = f"{query}_{k}_{score_threshold}_{str(filter_metadata)}"
                if use_cache and cache_key in self.query_cache:
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return self.query_cache[cache_key]
                
                # Create query embedding
                start_time = datetime.now()
                query_embedding = self.embedding_model.encode(
                    query, 
                    normalize_embeddings=True
                ).tolist()
                
                # Build filter if provided
                where_filter = None
                if filter_metadata:
                    where_filter = {"$and": []}
                    for key, value in filter_metadata.items():
                        if isinstance(value, list):
                            where_filter["$and"].append({key: {"$in": value}})
                        else:
                            where_filter["$and"].append({key: value})
                
                # Query with more results for filtering
                n_results = min(k * 3, self.collection.count())
                if n_results == 0:
                    return []
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                
                similar_docs = []
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i]
                    document = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    
                    # Validate distance
                    if distance is None or (isinstance(distance, float) and 
                                           (math.isnan(distance) or math.isinf(distance))):
                        distance = 1.0
                    
                    score = 1.0 - float(distance)
                    
                    # Apply threshold
                    if score < score_threshold:
                        continue
                    
                    # Clamp score
                    score = max(0.0, min(1.0, score))
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(score, metadata)
                    
                    similar_docs.append({
                        "text": document,
                        "metadata": metadata,
                        "distance": float(distance),
                        "score": float(score),
                        "confidence": confidence,
                        "id": results["ids"][0][i] if results["ids"] else f"doc_{i}"
                    })
                
                # Sort by score (highest first) and take top k
                similar_docs.sort(key=lambda x: x['score'], reverse=True)
                similar_docs = similar_docs[:k]
                
                # Add query time to metadata
                query_time = (datetime.now() - start_time).total_seconds()
                for doc in similar_docs:
                    doc['metadata']['_query_time'] = query_time
                    doc['metadata']['_query'] = query[:100]  # Store first 100 chars
                
                # Cache results
                if use_cache and similar_docs:
                    self.query_cache[cache_key] = similar_docs
                
                logger.debug(f"Search completed in {query_time:.3f}s: "
                           f"found {len(similar_docs)} results")
                
                return similar_docs
                
            except Exception as e:
                logger.error(f"Error searching: {e}")
                self.metrics['errors'] += 1
                return []
    
    def _calculate_confidence(self, score: float, metadata: Dict) -> float:
        """Calculate confidence score based on relevance and metadata quality"""
        confidence = score  # Start with relevance score
        
        # Boost based on metadata completeness
        required_fields = ['source', 'source_type', 'title']
        present_fields = sum(1 for field in required_fields if field in metadata and metadata[field])
        metadata_boost = present_fields / len(required_fields) * 0.1  # Up to 10% boost
        
        # Boost based on content length (if available)
        if 'content_length' in metadata:
            length = metadata['content_length']
            if isinstance(length, (int, float)):
                if length > 1000:
                    length_boost = 0.05  # Longer content more reliable
                else:
                    length_boost = -0.05  # Shorter content less reliable
                metadata_boost += length_boost
        
        confidence += metadata_boost
        return min(1.0, max(0.0, confidence))
    
    def semantic_search(self, 
                       queries: List[str], 
                       k: int = 5,
                       combine_results: bool = True) -> Union[List[Dict], List[List[Dict]]]:
        """Perform semantic search for multiple queries"""
        all_results = []
        
        for query in queries:
            results = self.search_similar(query, k=k, use_cache=False)
            all_results.append(results)
        
        if combine_results and len(queries) > 1:
            # Combine and deduplicate results from multiple queries
            combined = []
            seen_ids = set()
            
            for results in all_results:
                for result in results:
                    doc_id = result.get('id', hash(result['text']))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        combined.append(result)
            
            # Sort by average score if from multiple queries
            combined.sort(key=lambda x: x['score'], reverse=True)
            return combined[:k]
        
        return all_results if len(queries) > 1 else all_results[0]
    
    def clear_embeddings_cache(self):
        """Clear embedding model cache to free memory"""
        try:
            if hasattr(self.embedding_model, 'model'):
                if TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                gc.collect()
                logger.debug("Cleared embeddings cache")
        except Exception as e:
            logger.warning(f"Could not clear embeddings cache: {e}")
    
    # Helper methods for specific document types
    def add_web_documents(self, web_docs: List[Dict], **kwargs) -> Dict[str, Any]:
        """Thêm các web documents (từ crawler)"""
        chunks = []
        for doc in web_docs:
            chunk = {
                'text': doc.get('content', ''),
                'metadata': {
                    'source_type': 'web',
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'source': 'web_crawler',
                    'crawled_at': doc.get('crawled_at', datetime.now().isoformat()),
                    'content_length': len(doc.get('content', '')),
                    'depth': doc.get('depth', 0),
                    'importance_score': doc.get('importance_score', 0),
                    'domain': doc.get('domain', ''),
                    'language': doc.get('language', 'vi')
                }
            }
            chunks.append(chunk)
        
        return self.add_documents(chunks, **kwargs)
    
    def add_document_chunks(self, doc_chunks: List[Dict], **kwargs) -> Dict[str, Any]:
        """Thêm các document chunks (PDF, Word, etc.)"""
        chunks = []
        for chunk in doc_chunks:
            metadata = chunk.get('metadata', {})
            if 'source_type' not in metadata:
                metadata['source_type'] = 'document'
            
            # Extract filename if available
            if 'file_path' in metadata:
                metadata['filename'] = os.path.basename(metadata['file_path'])
            
            enhanced_chunk = {
                'text': chunk.get('text', ''),
                'metadata': metadata
            }
            chunks.append(enhanced_chunk)
        
        return self.add_documents(chunks, **kwargs)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Lấy thống kê chi tiết về collection"""
        with self.lock:
            try:
                count = self.collection.count()
                
                if count == 0:
                    return {
                        'total_documents': 0,
                        'collection_name': self.collection_name,
                        'embedding_model': self.embedding_model_name,
                        'status': 'empty'
                    }
                
                # Sample documents for analysis
                sample_size = min(1000, count)
                sample = self.collection.peek(limit=sample_size)
                
                sources = defaultdict(int)
                source_types = defaultdict(int)
                content_lengths = []
                
                for metadata in sample.get('metadatas', []):
                    source = metadata.get('source', 'unknown')
                    source_type = metadata.get('source_type', 'unknown')
                    
                    sources[source] += 1
                    source_types[source_type] += 1
                    
                    # Estimate content length
                    content_length = metadata.get('content_length', 0)
                    if isinstance(content_length, (int, float)):
                        content_lengths.append(content_length)
                
                # Calculate statistics
                avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
                
                stats = {
                    'total_documents': count,
                    'processed_urls_count': len(self.processed_urls),
                    'processed_files_count': len(self.processed_files),
                    'unique_content_hashes': len(self.content_hashes),
                    'sources_distribution': dict(sources),
                    'source_types_distribution': dict(source_types),
                    'average_content_length': avg_content_length,
                    'embedding_model': self.embedding_model_name,
                    'collection_name': self.collection_name,
                    'persist_directory': str(self.persist_directory),
                    'metrics': self.metrics.copy(),
                    'cache_size': len(self.query_cache),
                    'collection_size_limit': self.max_collection_size,
                    'current_size_percentage': (count / self.max_collection_size * 100) if self.max_collection_size > 0 else 0
                }
                
                return stats
                
            except Exception as e:
                logger.error(f"Error getting collection stats: {e}")
                return {'error': str(e), 'error_type': type(e).__name__}
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Lấy document bằng ID"""
        with self.lock:
            try:
                results = self.collection.get(ids=[doc_id])
                if results['documents']:
                    return {
                        'found': True,
                        'document': results['documents'][0],
                        'metadata': results['metadatas'][0] if results['metadatas'] else {},
                        'id': doc_id,
                        'source_info': self.id_to_source.get(doc_id, {})
                    }
                return {'found': False, 'error': 'Document not found'}
            except Exception as e:
                return {'found': False, 'error': str(e)}
    
    def get_document_by_url(self, url: str) -> Dict[str, Any]:
        """Tìm document bằng URL"""
        with self.lock:
            try:
                doc_id = self.url_to_id.get(url)
                if not doc_id:
                    return {'found': False, 'error': 'URL not found'}
                return self.get_document_by_id(doc_id)
            except Exception as e:
                return {'found': False, 'error': str(e)}
    
    def update_document(self, doc_id: str, content: str, metadata: Dict = None) -> Dict[str, Any]:
        """Cập nhật document bằng ID"""
        with self.lock:
            try:
                # Get existing document
                existing = self.collection.get(ids=[doc_id])
                if not existing['documents']:
                    return {'success': False, 'error': 'Document not found'}
                
                # Prepare new metadata
                new_metadata = existing['metadatas'][0] if existing['metadatas'] else {}
                if metadata:
                    new_metadata.update(metadata)
                
                new_metadata['updated_at'] = datetime.now().isoformat()
                new_metadata['update_count'] = new_metadata.get('update_count', 0) + 1
                
                # Create new embedding
                new_embedding = self.embedding_model.encode(
                    content, 
                    normalize_embeddings=True
                ).tolist()
                
                # Update in collection
                self.collection.update(
                    embeddings=[new_embedding],
                    documents=[content],
                    metadatas=[new_metadata],
                    ids=[doc_id]
                )
                
                # Update content hash
                self.content_hashes[doc_id] = (self.get_content_hash(content), datetime.now())
                
                # Clear cache
                self.query_cache.clear()
                
                logger.info(f"Updated document: {doc_id}")
                return {'success': True, 'id': doc_id}
                
            except Exception as e:
                logger.error(f"Error updating document: {e}")
                return {'success': False, 'error': str(e)}
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Xóa document bằng ID"""
        with self.lock:
            try:
                # Remove from collection
                self.collection.delete(ids=[doc_id])
                
                # Clean up tracking data
                if doc_id in self.content_hashes:
                    del self.content_hashes[doc_id]
                if doc_id in self.id_to_source:
                    # Also remove from URL/file mappings
                    source_info = self.id_to_source[doc_id]
                    identifier = source_info.get('identifier', '')
                    source_type = source_info.get('source_type', '')
                    
                    if source_type == 'web' and identifier in self.url_to_id:
                        del self.url_to_id[identifier]
                        self.processed_urls.discard(identifier)
                    elif source_type == 'document' and identifier in self.file_to_id:
                        del self.file_to_id[identifier]
                        self.processed_files.discard(identifier)
                    
                    del self.id_to_source[doc_id]
                
                # Clear cache
                self.query_cache.clear()
                
                logger.info(f"Deleted document: {doc_id}")
                return {'success': True, 'id': doc_id}
                
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                return {'success': False, 'error': str(e)}
    
    def clear_collection(self, confirm: bool = False) -> Dict[str, Any]:
        """Xóa toàn bộ collection và reset tracking"""
        if not confirm:
            return {
                'status': 'confirmation_required',
                'message': 'Please set confirm=True to clear the collection'
            }
        
        with self.lock:
            try:
                # Backup if enabled
                if self.enable_backup:
                    self._auto_backup(force=True)
                
                # Delete collection
                self.client.delete_collection(self.collection_name)
                
                # Recreate collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "created_at": datetime.now().isoformat(),
                        "embedding_model": self.embedding_model_name,
                        "reset_count": self.metrics.get('reset_count', 0) + 1
                    }
                )
                
                # Reset tracking
                self.processed_urls.clear()
                self.processed_files.clear()
                self.content_hashes.clear()
                self.url_to_id.clear()
                self.file_to_id.clear()
                self.id_to_source.clear()
                self.query_cache.clear()
                
                # Update metrics
                self.metrics['reset_count'] = self.metrics.get('reset_count', 0) + 1
                self.metrics['collection_size'] = 0
                
                # Remove processed data file
                processed_file = self.persist_directory / "processed_data.json"
                if processed_file.exists():
                    processed_file.unlink()
                
                # Clear embeddings cache
                self.clear_embeddings_cache()
                
                logger.info("Collection cleared and tracking reset")
                return {
                    'status': 'success',
                    'message': 'Collection cleared',
                    'collection_name': self.collection_name
                }
                
            except Exception as e:
                logger.error(f"Error clearing collection: {e}")
                return {'status': 'error', 'error': str(e)}
    
    def _auto_backup(self, force: bool = False):
        """Tự động backup collection"""
        if not self.enable_backup:
            return
        
        try:
            # Check if backup is needed (once per day)
            last_backup = self.metrics.get('last_backup')
            if not force and last_backup:
                last_date = datetime.fromisoformat(last_backup)
                if (datetime.now() - last_date).days < 1:
                    return
            
            backup_dir = self.persist_directory / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            
            # Copy entire ChromaDB directory
            shutil.copytree(self.persist_directory, backup_path)
            
            # Also backup processed data
            processed_backup = backup_path / "processed_data.json"
            if (self.persist_directory / "processed_data.json").exists():
                shutil.copy2(
                    self.persist_directory / "processed_data.json",
                    processed_backup
                )
            
            self.metrics['last_backup'] = datetime.now().isoformat()
            logger.info(f"Backup created: {backup_path}")
            
            # Clean old backups (keep last 7)
            backups = sorted(backup_dir.glob("backup_*"))
            for old_backup in backups[:-7]:
                shutil.rmtree(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def export_collection(self, export_path: str = None) -> Dict[str, Any]:
        """Export collection to JSON format"""
        try:
            if export_path is None:
                export_path = self.persist_directory / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                export_path = Path(export_path)
            
            # Get all documents
            all_docs = self.collection.get()
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'collection_name': self.collection_name,
                    'total_documents': len(all_docs['ids']),
                    'embedding_model': self.embedding_model_name,
                    'version': '1.0'
                },
                'documents': []
            }
            
            for i in range(len(all_docs['ids'])):
                doc_data = {
                    'id': all_docs['ids'][i],
                    'text': all_docs['documents'][i] if all_docs['documents'] else '',
                    'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {},
                    'embedding': all_docs['embeddings'][i] if all_docs['embeddings'] else None
                }
                export_data['documents'].append(doc_data)
            
            # Save to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Exported collection to {export_path}")
            return {
                'success': True,
                'export_path': str(export_path),
                'document_count': len(all_docs['ids'])
            }
            
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return {'success': False, 'error': str(e)}
    
    def as_retriever(self, search_kwargs=None):
        """Biến vector store tương thích với LangChain retriever"""
        search_kwargs = search_kwargs or {"k": 5, "score_threshold": 0.7}
        
        class VectorStoreRetriever:
            def __init__(self, store, kwargs):
                self.store = store
                self.kwargs = kwargs
                self.name = "EnhancedVectorStoreRetriever"
                self.description = "Retriever for EnhancedVectorStore with semantic search"
            
            def get_relevant_documents(self, query: str) -> List[Any]:
                results = self.store.search_similar(
                    query, 
                    k=self.kwargs.get("k", 5),
                    score_threshold=self.kwargs.get("score_threshold", 0.7),
                    filter_metadata=self.kwargs.get("filter", None),
                    use_cache=self.kwargs.get("use_cache", True)
                )
                
                documents = []
                for result in results:
                    # Create document object compatible with LangChain
                    class Document:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                        
                        def to_dict(self):
                            return {
                                'page_content': self.page_content,
                                'metadata': self.metadata
                            }
                    
                    doc = Document(
                        page_content=result['text'],
                        metadata=result['metadata']
                    )
                    
                    # Add additional information as attributes
                    for key in ['score', 'distance', 'confidence', 'id']:
                        if key in result:
                            setattr(doc, key, result[key])
                    
                    documents.append(doc)
                return documents
            
            def invoke(self, query: str) -> List[Any]:
                """Alias for get_relevant_documents for newer LangChain versions"""
                return self.get_relevant_documents(query)
        
        return VectorStoreRetriever(self, search_kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái health của vector store"""
        try:
            # Check ChromaDB connection
            collection_count = self.collection.count()
            
            # Check embedding model
            test_embedding = self.embedding_model.encode("test")
            
            # Check storage
            storage_available = self.persist_directory.exists() and os.access(self.persist_directory, os.W_OK)
            
            # Calculate uptime metrics
            status = {
                'status': 'healthy',
                'collection_count': collection_count,
                'embedding_model': 'working',
                'storage': 'available' if storage_available else 'unavailable',
                'cache_size': len(self.query_cache),
                'processed_urls': len(self.processed_urls),
                'processed_files': len(self.processed_files),
                'unique_hashes': len(self.content_hashes),
                'last_backup': self.metrics.get('last_backup'),
                'total_searches': self.metrics['search_count'],
                'total_errors': self.metrics['errors'],
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup khi object bị destroy"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Save processed data one last time
        try:
            self._save_processed_data()
        except:
            pass
        
        # Clear cache
        self.clear_embeddings_cache()


# Factory function for backward compatibility
def create_vector_store(persist_directory: str = "./chroma_db",
                       collection_name: str = "arbin_documents",
                       embedding_model: str = "all-MiniLM-L6-v2",
                       **kwargs) -> EnhancedVectorStore:
    """Factory function to create EnhancedVectorStore"""
    return EnhancedVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
        **kwargs
    )


# Backward compatibility
VectorStore = EnhancedVectorStore