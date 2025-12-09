#!/usr/bin/env python3
"""
🚀 RUN FULL RAG PIPELINE - SIMPLEST VERSION
"""

import os
import sys
import subprocess

def run_script(script_name):
    """Run a script and return True if successful"""
    if not os.path.exists(script_name):
        print(f"❌ Script not found: {script_name}")
        return False
    
    print(f"\n{'='*50}")
    print(f"🚀 Running: {script_name}")
    print(f"{'='*50}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name])
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 Starting full RAG pipeline...")
    
    # Step 1: Ingest local docs
    run_script("script_1_ingest_local.py")
    
    # Step 2: Web crawl
    run_script("script_2_web_crawl.py")
    
    # Step 3: Clean & chunk
    run_script("script_3_clean_chunk.py")
    
    # Step 4: Embed & store
    run_script("script_4_embed_store.py")
    
    # Step 5: Test retrieval
    run_script("script_5_test_retrieval.py")
    
    print("\n" + "="*50)
    print("✅ Pipeline completed!")
    print("="*50)

if __name__ == "__main__":
    main()