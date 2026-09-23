# RAG Demo v2 — প্রোডাকশনের দিকে ৮টা ধাপ

## ফাইল স্ট্রাকচার
```
rag_demo_v2/
├── rag_core.py       # মূল লজিক: load, chunk, embed, index, retrieve, generate
├── rag_pipeline.py   # CLI (টার্মিনাল থেকে চালানোর জন্য)
├── api.py            # FastAPI ওয়েব সার্ভার (/ask, /health)
├── static/index.html # সাধারণ চ্যাট UI
├── eval.py           # টেস্ট/ইভ্যালুয়েশন স্ক্রিপ্ট
├── requirements.txt
├── .gitignore
└── docs/             # <-- আপনার .txt/.pdf ফাইলগুলো এখানে রাখুন
```

চালানোর আগে `docs/` ফোল্ডার বানিয়ে তার ভেতর আপনার নোট/ডকুমেন্ট ফাইলগুলো রাখুন
(যেমন `ai_bistarito_note.txt`), এবং `.env`-এ `GEMINI_API_KEY=...` বসান।

```bash
pip install -r requirements.txt

# CLI চালাতে:
python rag_pipeline.py

# ওয়েব API + UI চালাতে:
uvicorn api:app --reload
# তারপর ব্রাউজারে: http://127.0.0.1:8000
```

## ৮টা ধাপ কীভাবে কভার হলো

| # | কাজ | কোথায় |
|---|-----|--------|
| ১ | Index persist ও incremental update | `save_vector_store` / `load_vector_store` / `add_document_to_store` |
| ২ | Sentence-boundary chunking + multi-doc | `chunk_text` (paragraph→sentence aware) + `load_documents_from_folder` |
| ৩ | Error handling, retry, logging | সব ফাংশনে try/except + `logging` মডিউল (`rag_pipeline.log`) |
| ৪ | CLI → API + UI | `api.py` (FastAPI) + `static/index.html` |
| ৫ | Security | `.gitignore` (`.env` বাদ), `sanitize_query()`, system prompt আলাদা parameter, per-IP rate limit |
| ৬ | Retrieval ও answer quality | `retrieve_relevant_chunks` (FAISS + BM25 hybrid → RRF → cross-encoder reranker দিয়ে চূড়ান্ত বাছাই) + প্রতিটা উত্তরে source citation |
| ৭ | Testing ও evaluation | `eval.py` — retrieval hit rate, keyword recall, groundedness/hallucination চেক |
| ৮ | Deployment, স্কেলিং, কস্ট মনিটরিং | নিচে দেখুন |

## ধাপ ৮ বিস্তারিত: Deployment ও মনিটরিং

**ডেটা বাড়লে:** `IndexFlatIP` (brute-force) কয়েক হাজার চাঙ্ক পর্যন্ত ঠিক আছে।
এরপর দরকার হলে managed vector DB-তে সরান — Chroma (self-host, সহজ),
Qdrant (self-host/cloud), বা Pinecone (fully managed)। এই কোডে শুধু
`build_vector_store`/`retrieve_relevant_chunks` ফাংশন দুটো বদলালেই হবে,
বাকি সব একই থাকবে।

**ডিপ্লয় করতে:**
- ছোট প্রোজেক্টের জন্য Render বা Railway-তে `uvicorn api:app` চালানো সহজ।
- `embedding_model` লোড হতে সময় লাগে, তাই কোল্ড-স্টার্ট এড়াতে "always on" instance ভালো।
- `.env`-এর `GEMINI_API_KEY` কখনো কোডে হার্ডকোড না করে হোস্টিং প্ল্যাটফর্মের
  secret/environment variable সেকশনে বসান।

**খরচ ও ব্যবহার মনিটর করতে:**
- `api.py`-তে ইতিমধ্যে per-IP rate limit (প্রতি মিনিটে ১০টা রিকোয়েস্ট) আছে —
  স্কেল অনুযায়ী `RATE_LIMIT`/`RATE_WINDOW` বদলান।
- প্রতিটা রিকোয়েস্ট `logger.info` দিয়ে লগ হচ্ছে (`rag_pipeline.log`) — চাইলে
  এই লগ থেকে দৈনিক রিকোয়েস্ট সংখ্যা গুনে Gemini-এর free-tier কোটার সাথে তুলনা করুন।
- বড় স্কেলে গেলে Prometheus/Grafana বা হোস্টিং প্ল্যাটফর্মের বিল্ট-ইন মেট্রিক্স ব্যবহার করুন।
