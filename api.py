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
import hmac
import threading
from collections import defaultdict, deque

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, Header
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_core import (
    logger, get_or_build_index, answer_query, sanitize_query,
    create_chat_session, RagSession, stream_answer_query,
    Chunk, DOCS_FOLDER, load_document, chunk_file, create_embeddings,
    save_vector_store,
)

app = FastAPI(title="RAG API")
from orders_api import router as orders_router
app.include_router(orders_router)
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


@app.get("/report")
def serve_report():
    return FileResponse("static/report.html")


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
        if result.get("not_found"):
            _log_unanswered(query)
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
                if event.get("type") == "done" and event.get("not_found"):
                    _log_unanswered(query)
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
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # ৫০ MB
ALLOWED_EXTENSIONS = (".txt", ".pdf", ".docx")
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
    texts = chunk_file(filename, text)
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


def _remove_document_atomically(filename: str) -> int:
    """
    #Document Delete: একটা ডকুমেন্টের সব চাঙ্ক ইনডেক্স থেকে সরায়।
    FAISS flat ইনডেক্স থেকে সরাসরি মোছা যায় না, তাই বাকি ভেক্টরগুলো (reconstruct করে, নতুন করে
    embed না করে — ফলে API খরচ নেই) দিয়ে নতুন ইনডেক্স বানানো হয়, ডিস্কে সেভ হয়, তারপর মেমরিতে বদলানো হয়।
    সেভ ব্যর্থ হলে এক্সেপশন উঠে যায় আর মেমরির INDEX/CHUNKS অক্ষত থাকে।
    """
    global INDEX, CHUNKS
    if INDEX.ntotal != len(CHUNKS):
        raise RuntimeError(f"ইনডেক্স ({INDEX.ntotal}) আর চাঙ্ক লিস্ট ({len(CHUNKS)}) মিলছে না — মোছা বন্ধ করা হলো।")

    keep = [i for i, c in enumerate(CHUNKS) if c.source != filename]
    removed = len(CHUNKS) - len(keep)

    new_index = faiss.IndexFlatIP(INDEX.d)
    if keep:
        vectors = INDEX.reconstruct_n(0, INDEX.ntotal)
        new_index.add(np.ascontiguousarray(vectors[keep], dtype="float32"))
    new_chunks = [CHUNKS[i] for i in keep]

    save_vector_store(new_index, new_chunks)
    # দুটো বদলের মাঝে মাইক্রোসেকেন্ডের একটা ফাঁক থাকে; ছোট/মাঝারি ডেমোর জন্য এটা গ্রহণযোগ্য
    CHUNKS = new_chunks
    INDEX = new_index
    return removed


# ---------------------------------------------------------
# Admin অথেন্টিকেশন: /upload আর DELETE /documents শুধু ADMIN_KEY জানা লোকের জন্য
# ক্লায়েন্ট হেডার পাঠায়:  X-Admin-Key: <আপনার ADMIN_KEY>
# ADMIN_KEY সেট না থাকলে দুটো এন্ডপয়েন্টই বন্ধ (fail closed), যাতে ভুল করে খোলা না থাকে।
# ---------------------------------------------------------
ADMIN_KEY = os.getenv("ADMIN_KEY", "")


def require_admin(request: Request, x_admin_key: str = Header(default="")):
    client_ip = request.client.host if request.client else "unknown"
    if not ADMIN_KEY:
        logger.error("ADMIN_KEY সেট করা নেই — আপলোড/ডিলিট বন্ধ রাখা হলো।")
        raise HTTPException(status_code=503, detail="সার্ভারে ADMIN_KEY সেট করা নেই, তাই এই কাজ বন্ধ আছে।")
    if not hmac.compare_digest(x_admin_key.encode("utf-8"), ADMIN_KEY.encode("utf-8")):
        logger.warning(f"🚫 ভুল/খালি admin key ({request.method} {request.url.path}) IP: {client_ip}")
        check_rate_limit(client_ip)  # ভুল চেষ্টা বেশি হলে 429 — key আন্দাজ করা কঠিন করে
        raise HTTPException(status_code=401, detail="অনুমতি নেই — সঠিক admin key দিন।")


@app.delete("/documents/{filename:path}", dependencies=[Depends(require_admin)])
def delete_document(filename: str, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    with _upload_lock:
        if not any(c.source == filename for c in CHUNKS):
            raise HTTPException(status_code=404, detail=f"'{filename}' নামে কোনো ইনডেক্স করা ডকুমেন্ট নেই।")

        try:
            removed = _remove_document_atomically(filename)
        except Exception as e:
            logger.exception(f"/documents DELETE এন্ডপয়েন্টে সমস্যা ({filename}): {e}")
            raise HTTPException(status_code=500, detail="ডকুমেন্ট মুছতে গিয়ে সমস্যা হয়েছে।")

        # docs/ ফোল্ডার থেকেও ফাইলটা সরানো হয়, নইলে সার্ভার রিস্টার্টে ইনডেক্স নতুন করে বানালে ফিরে আসত
        docs_root = os.path.realpath(DOCS_FOLDER)
        path = os.path.realpath(os.path.join(DOCS_FOLDER, os.path.basename(filename)))
        if os.path.dirname(path) == docs_root:
            _remove_quietly(path)

    logger.info(f"🗑️ ডকুমেন্ট মোছা হলো: {filename} ({removed}টা চাঙ্ক সরানো, বাকি {len(CHUNKS)}টা)")
    return {"deleted": filename, "chunks_removed": removed, "total_chunks": len(CHUNKS)}


@app.post("/upload", dependencies=[Depends(require_admin)])
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


# ---------------------------------------------------------
# #Handoff: বট উত্তর দিতে না পারলে কাস্টমার ফোন নম্বর রেখে যায়, দোকানের লোক পরে যোগাযোগ করে।
# ডেটা handoffs.json ফাইলে জমা হয় (DATA_DIR env দিলে সেই ফোল্ডারে)।
# ---------------------------------------------------------
HANDOFF_FILE = os.path.join(os.getenv("DATA_DIR", "."), "handoffs.json")
_handoff_lock = threading.Lock()
_PHONE_OK = re.compile(r"^[0-9+\-\s()]{6,20}$")


class HandoffRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    phone: str = Field(..., min_length=6, max_length=20)


def _read_handoffs() -> list:
    if not os.path.exists(HANDOFF_FILE):
        return []
    try:
        with open(HANDOFF_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("handoffs.json পড়তে সমস্যা, খালি তালিকা ধরা হলো")
        return []


@app.post("/handoff")
def create_handoff(req: HandoffRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    # বাংলা সংখ্যা (০১৭১...) লিখলেও চলবে: ইংরেজি সংখ্যায় বদলে নেওয়া হয়
    phone = req.phone.strip().translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))
    if not _PHONE_OK.match(phone) or sum(ch.isdigit() for ch in phone) < 6:
        raise HTTPException(status_code=400, detail="ফোন নম্বরটা ঠিক মনে হচ্ছে না, আরেকবার দেখে লিখুন।")

    with _handoff_lock:
        items = _read_handoffs()
        items.append({
            "id": len(items) + 1,
            "question": sanitize_query(req.question),
            "phone": phone,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done": False,
        })
        tmp = HANDOFF_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        os.replace(tmp, HANDOFF_FILE)  # মাঝপথে বন্ধ হলেও ফাইল নষ্ট হয় না
    logger.info(f"📞 নতুন handoff অনুরোধ (#{len(items)})")
    return {"status": "saved"}


@app.get("/admin/handoffs", dependencies=[Depends(require_admin)])
def list_handoffs():
    with _handoff_lock:
        items = _read_handoffs()
    return {"handoffs": list(reversed(items))}


# ---------------------------------------------------------
# #Report: দোকানদারের রিপোর্ট পেজের জন্য
#   - ফোন অনুরোধে "যোগাযোগ হয়েছে" চিহ্ন দেওয়া
#   - বট যে প্রশ্নের উত্তর দিতে পারেনি তার খাতা (unanswered.json) — ফোন নম্বর ছাড়াই
# ---------------------------------------------------------
UNANSWERED_FILE = os.path.join(os.getenv("DATA_DIR", "."), "unanswered.json")
MAX_UNANSWERED = 500          # ফাইল যেন অসীম বড় না হয়, তাই পুরনোগুলো বাদ পড়ে
_unanswered_lock = threading.Lock()


class HandoffPatch(BaseModel):
    done: bool


def _write_json_atomic(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_unanswered() -> list:
    if not os.path.exists(UNANSWERED_FILE):
        return []
    try:
        with open(UNANSWERED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("unanswered.json পড়তে সমস্যা, খালি তালিকা ধরা হলো")
        return []


def _log_unanswered(question: str):
    """একই প্রশ্ন আবার এলে নতুন সারি না বানিয়ে গুনতি বাড়ায়। এখানে ব্যর্থ হলে চ্যাট যেন না থামে।"""
    try:
        q = sanitize_query(question)
        key = " ".join(q.lower().split())
        if not key:
            return
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with _unanswered_lock:
            items = _read_unanswered()
            hit = next((it for it in items if " ".join(it["question"].lower().split()) == key), None)
            if hit:
                hit["count"] += 1
                hit["last"] = now
            else:
                next_id = max([it["id"] for it in items] + [0]) + 1
                items.append({"id": next_id, "question": q, "count": 1, "first": now, "last": now})
                items = items[-MAX_UNANSWERED:]
            _write_json_atomic(UNANSWERED_FILE, items)
    except Exception:
        logger.exception("unanswered খাতায় লিখতে সমস্যা (চ্যাট চলছে)")


@app.patch("/admin/handoffs/{handoff_id}", dependencies=[Depends(require_admin)])
def update_handoff(handoff_id: int, req: HandoffPatch):
    with _handoff_lock:
        items = _read_handoffs()
        item = next((x for x in items if x.get("id") == handoff_id), None)
        if not item:
            raise HTTPException(status_code=404, detail="এই অনুরোধটা খুঁজে পাওয়া যায়নি।")
        item["done"] = req.done
        _write_json_atomic(HANDOFF_FILE, items)
    return item


@app.get("/admin/unanswered", dependencies=[Depends(require_admin)])
def list_unanswered():
    with _unanswered_lock:
        items = _read_unanswered()
    items.sort(key=lambda x: (x["count"], x["last"]), reverse=True)
    return {"unanswered": items}


@app.delete("/admin/unanswered/{item_id}", dependencies=[Depends(require_admin)])
def delete_unanswered(item_id: int):
    with _unanswered_lock:
        items = _read_unanswered()
        left = [x for x in items if x.get("id") != item_id]
        if len(left) == len(items):
            raise HTTPException(status_code=404, detail="এই প্রশ্নটা খুঁজে পাওয়া যায়নি।")
        _write_json_atomic(UNANSWERED_FILE, left)
    return {"deleted": item_id}
