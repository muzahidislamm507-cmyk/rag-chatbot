"""
FastAPI দিয়ে RAG বটকে একটা ওয়েব API + সাধারণ চ্যাট UI বানানো হলো (#৪)।

চালানোর নিয়ম:
    uvicorn api:app --reload
তারপর ব্রাউজারে যান: http://127.0.0.1:8000

এই ফাইলে যোগ করা হয়েছে:
    - /ask   : মূল প্রশ্ন-উত্তর এন্ডপয়েন্ট
    - /health: সার্ভার বেঁচে আছে কিনা চেক করার এন্ডপয়েন্ট
    - সাধারণ in-memory rate limiting (#৮ — হঠাৎ বিল বেড়ে যাওয়া ঠেকাতে)
    - প্রতিটা রিকোয়েস্টের usage logging (#৮ — কে কতবার জিজ্ঞেস করলো তার ট্র্যাক)
"""

import os
import re
import time
import uuid
import json
import threading
from collections import defaultdict, deque

import faiss
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_core import (
    logger, get_or_build_index, answer_query, sanitize_query,
    create_chat_session, RagSession, stream_answer_query,
    Chunk, DOCS_FOLDER, load_document, chunk_text, create_embeddings,
    save_vector_store,
)

app = FastAPI(title="RAG API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# স্টার্টআপেই ইনডেক্স লোড/বানিয়ে মেমরিতে রাখা হয়, প্রতি রিকোয়েস্টে না
INDEX, CHUNKS = None, None

# #Conversation Memory: প্রতিটা ব্রাউজার সেশনের জন্য আলাদা chat history রাখা হয়।
# ছোট/মাঝারি ডেমোর জন্য এই in-memory dict যথেষ্ট; বড় স্কেলে (একাধিক সার্ভার
# ইনস্ট্যান্স) গেলে Redis-এর মতো শেয়ার্ড স্টোরে সরাতে হবে।
SESSIONS: dict[str, RagSession] = {}
SESSION_TTL = 60 * 30  # ৩০ মিনিট নিষ্ক্রিয় থাকলে সেশন মুছে ফেলা হয়
_session_last_used: dict[str, float] = {}


def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, t in _session_last_used.items() if now - t > SESSION_TTL]
    for sid in expired:
        SESSIONS.pop(sid, None)
        _session_last_used.pop(sid, None)


@app.on_event("startup")
def startup():
    global INDEX, CHUNKS
    INDEX, CHUNKS = get_or_build_index()
    logger.info("🚀 API স্টার্টআপ সম্পন্ন, ইনডেক্স রেডি।")


# ---------------------------------------------------------
# #৮ সাধারণ Rate Limiting (per-IP, in-memory)
# প্রোডাকশনে বড় স্কেলে slowapi/Redis ব্যবহার করা ভালো, কিন্তু
# ছোট/মাঝারি ডেমোর জন্য এই সাধারণ token-bucket যথেষ্ট।
# ---------------------------------------------------------
RATE_LIMIT = 10          # প্রতি WINDOW সেকেন্ডে সর্বোচ্চ কতবার
RATE_WINDOW = 60
_request_log: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client_ip: str):
    now = time.time()
    q = _request_log[client_ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="অনেক বেশি রিকোয়েস্ট হয়ে গেছে, একটু পর আবার চেষ্টা করুন।")
    q.append(now)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None  # #Conversation Memory: প্রথম রিকোয়েস্টে None, সার্ভার একটা বানিয়ে ফেরত দেয়


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(CHUNKS) if CHUNKS else 0, "active_sessions": len(SESSIONS)}


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    query = sanitize_query(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="প্রশ্ন খালি রাখা যাবে না।")

    _cleanup_expired_sessions()

    # সেশন খুঁজে বের করা বা নতুন বানানো
    session_id = req.session_id
    if not session_id or session_id not in SESSIONS:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = create_chat_session()
        logger.info(f"🆕 নতুন সেশন তৈরি হলো: {session_id[:8]}...")
    _session_last_used[session_id] = time.time()

    logger.info(f"📥 API রিকোয়েস্ট থেকে ({client_ip}, session {session_id[:8]}): {query[:80]}")

    try:
        result = answer_query(query, INDEX, CHUNKS, session=SESSIONS[session_id])
        result["session_id"] = session_id
        return result
    except Exception as e:
        logger.exception(f"/ask এন্ডপয়েন্টে সমস্যা: {e}")
        raise HTTPException(status_code=500, detail="সার্ভারে একটা সমস্যা হয়েছে।")


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request):
    """
    #Streaming Answer: উত্তর একসাথে না দিয়ে ধীরে ধীরে পাঠায়।
    প্রতিটা লাইন একটা JSON object (newline-delimited JSON) —
    {"type": "session", ...} → {"type": "chunk", "text": ...} (বহুবার) → {"type": "done", ...}
    """
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    query = sanitize_query(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="প্রশ্ন খালি রাখা যাবে না।")

    _cleanup_expired_sessions()

    session_id = req.session_id
    if not session_id or session_id not in SESSIONS:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = create_chat_session()
        logger.info(f"🆕 নতুন সেশন তৈরি হলো (stream): {session_id[:8]}...")
    _session_last_used[session_id] = time.time()

    logger.info(f"📥 Stream রিকোয়েস্ট ({client_ip}, session {session_id[:8]}): {query[:80]}")

    def event_generator():
        yield json.dumps({"type": "session", "session_id": session_id}, ensure_ascii=False) + "\n"
        try:
            for event in stream_answer_query(query, INDEX, CHUNKS, session=SESSIONS[session_id]):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.exception(f"/ask/stream এন্ডপয়েন্টে সমস্যা: {e}")
            yield json.dumps({"type": "error", "text": "সার্ভারে একটা সমস্যা হয়েছে।"}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/reset")
def reset_session(req: dict):
    """ইউজার চাইলে সেশন রিসেট করে নতুন কথোপকথন শুরু করতে পারে।"""
    session_id = req.get("session_id")
    if session_id in SESSIONS:
        SESSIONS.pop(session_id, None)
        _session_last_used.pop(session_id, None)
    return {"status": "reset"}


# ---------------------------------------------------------
# #Document Management: ইনডেক্স করা ডকুমেন্টের তালিকা + নতুন ফাইল আপলোড
# (static/index.html আগে থেকেই /documents আর /upload কল করছিল)
# ---------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # ১০ MB
ALLOWED_EXTENSIONS = (".txt", ".pdf")
_upload_lock = threading.Lock()       # একসাথে দুটো আপলোড ইনডেক্স বদলাতে পারবে না


def _safe_filename(raw_name: str) -> str:
    """পাথ-ট্র্যাভার্সাল ঠেকায়; বাংলা অক্ষর, ইংরেজি অক্ষর, সংখ্যা, স্পেস, '.', '-', '_' রাখে।"""
    name = os.path.basename((raw_name or "").replace("\\", "/"))
    name = re.sub(r"[^\w.\- \u0980-\u09FF]", "_", name).strip(" .")
    return name


@app.get("/documents")
def list_documents():
    counts: dict[str, int] = {}
    for c in (CHUNKS or []):
        counts[c.source] = counts.get(c.source, 0) + 1
    return {"documents": [{"name": n, "chunks": k} for n, k in counts.items()]}


def _add_document_atomically(path: str, filename: str) -> int:
    """
    নতুন ফাইলের চাঙ্ক বানিয়ে ইনডেক্সে যোগ করে, ডিস্কে সেভ করে, তারপর মেমরির INDEX/CHUNKS বদলায়।
    ইনডেক্সের একটা ক্লোনে কাজ করে শেষে রেফারেন্স বদলানো হয় — ফলে চলমান কোনো প্রশ্নের সার্চ
    (FAISS add/search একসাথে thread-safe না) মাঝপথে ভেঙে পড়ে না। CHUNKS আগে, INDEX পরে বদলানো
    হয়, তাই ইনডেক্স কখনোই এমন চাঙ্কের দিকে ইশারা করে না যেটা লিস্টে নেই।
    """
    global INDEX, CHUNKS
    text = load_document(path)
    texts = chunk_text(text)
    if not texts:
        raise ValueError("ফাইলে কোনো পড়ার মতো টেক্সট পাওয়া যায়নি (স্ক্যান করা PDF হলে টেক্সট থাকে না)।")

    new_chunks = [Chunk(text=t, source=filename) for t in texts]
    embeddings = create_embeddings(texts)

    new_index = faiss.clone_index(INDEX)
    new_index.add(embeddings)
    all_chunks = CHUNKS + new_chunks

    save_vector_store(new_index, all_chunks)   # সেভ ব্যর্থ হলে এখানেই এক্সেপশন, মেমরি অক্ষত থাকে
    CHUNKS = all_chunks
    INDEX = new_index
    return len(new_chunks)


@app.post("/upload")
def upload_document(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    filename = _safe_filename(file.filename)
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="শুধু .txt বা .pdf ফাইল আপলোড করা যাবে।")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"ফাইল অনেক বড় (সর্বোচ্চ {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)।")
    if not data:
        raise HTTPException(status_code=400, detail="ফাইলটা খালি।")

    with _upload_lock:
        if any(c.source == filename for c in CHUNKS):
            raise HTTPException(status_code=409, detail=f"'{filename}' নামে ডকুমেন্ট আগেই ইনডেক্স করা আছে। নাম বদলে আবার চেষ্টা করুন।")

        os.makedirs(DOCS_FOLDER, exist_ok=True)
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "wb") as f:
            f.write(data)

        try:
            n = _add_document_atomically(path, filename)
        except (ValueError, UnicodeDecodeError) as e:
            _remove_quietly(path)
            msg = "টেক্সট ফাইলটা UTF-8 এনকোডিংয়ে সেভ করে আবার দিন।" if isinstance(e, UnicodeDecodeError) else str(e)
            raise HTTPException(status_code=422, detail=msg)
        except Exception as e:
            _remove_quietly(path)
            logger.exception(f"/upload এন্ডপয়েন্টে সমস্যা ({filename}): {e}")
            raise HTTPException(status_code=500, detail="ডকুমেন্ট ইনডেক্স করতে গিয়ে সমস্যা হয়েছে।")

    logger.info(f"📤 আপলোড সম্পন্ন: {filename} ({n}টা চাঙ্ক, মোট {len(CHUNKS)}টা)")
    return {"filename": filename, "chunks_added": n, "total_chunks": len(CHUNKS)}


def _remove_quietly(path: str):
    try:
        os.remove(path)
    except OSError:
        pass
