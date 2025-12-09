azvision-chatbot/
├── chroma_db/
├── documents/
├── data_layer/
│   ├── __init__.py
│   ├── web_crawler.py
│   ├── pdf_processor.py
│   ├── preprocessor.py
│   └── vector_store.py
├── ai_core/
│   ├── __init__.py
│   ├── llm_chain.py
│   ├── retrieval_qa.py
│   ├── parsers.py
│   ├── prompts.py
│   └── nlu_processor.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── components/
│       │   ├── Chat.jsx
│       │   ├── Header.jsx
│       │   └── InputArea.jsx
│       ├── App.jsx
│       ├── index.js
│       └── styles.css
├── .env.example
├── requirements.txt
├── main.py
└── README.md


# run block data layer
python -m data_layer.runs.script_1_ingest_local
python -m data_layer.runs.script_2_web_crawl
python -m data_layer.runs.run_3_preprocess_chunks
python -m data_layer.runs.run_4_embed_store
python -m data_layer.runs.run_5_search_demo
python -m data_layer.runs.run_6_merge_sources
python -m data_layer.runs.run_pipeline_full
python -m data_layer.runs.run_pipeline_full_plus