"""
chunker.py - গঠন-সচেতন (structure-aware) চাঙ্কিং (UrbanWear ডেমো ডক অনুযায়ী)

- Product catalog : একটা প্রোডাক্ট = একটা চাঙ্ক ("CURRENT SALE" আলাদা চাঙ্ক)
- FAQ             : একটা Q+A = একটা চাঙ্ক (ক্যাটাগরির নাম প্রিফিক্সে)
- Policy          : একটা নম্বরযুক্ত সেকশন = একটা চাঙ্ক
- Size guide      : একটা ALL-CAPS সেকশন = একটা চাঙ্ক
- অন্য ফাইল        : প্যারাগ্রাফ ধরে জোড়া

প্রতিটা চাঙ্কের শুরুতে: [Document Name > Section Title]

ব্যবহার:
    from chunker import chunk_document
    chunks = chunk_document("04-product-catalog.txt", text)
    # প্রতিটা chunk = {"text": ..., "metadata": {"doc", "section", "type"}}

টেস্ট:  python chunker.py docs/
"""
import os
import re
import sys
from string import capwords

MAX_CHARS = 1500

PRODUCT_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:Product\s*\d+\s*[:.\-–]\s*)?"
    r"(?P<title>[^\n]*?\(\s*[A-Z]{2,5}-\d+\s*\))\s*$"
)
FAQ_Q_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:Q\s*\d*\s*[:.)]|Question\s*\d*\s*[:.)]|প্রশ্ন\s*\d*\s*[:.)])\s*"
    r"(?P<q>.*)$",
    re.IGNORECASE,
)
POLICY_HEAD_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?P<num>\d+)\.\s+(?P<title>\S[^\n]*?)\s*$"
)


def _nice(s):
    """'RETURN AND REFUND POLICY' -> 'Return and Refund Policy'"""
    paren = re.search(r"\([^)]*\)", s)  # (T-shirts, hoodies) অংশ যেমন আছে তেমন থাকবে
    tail = paren.group(0) if paren else ""
    s = capwords(re.sub(r"\([^)]*\)", "", s).strip().lower())
    s = re.sub(r"\b(And|Of|The|For)\b", lambda m: m.group(1).lower(), s).replace("Faq", "FAQ")
    return f"{s} {tail}".strip()


def _is_caps_head(line):
    s = line.strip()
    if not s or len(s) > 60 or s.endswith((":", ".")):
        return False
    t = re.sub(r"\([^)]*\)", "", s).strip()  # (T-shirts, hoodies) অংশ বাদ
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 3 and t == t.upper()


def _is_policy_head(line):
    m = POLICY_HEAD_RE.match(line)
    if not m:
        return False
    title = m.group("title")
    return len(title) <= 80 and not title.endswith((".", ":", "।"))


def _doc_type(filename, lines):
    name = os.path.basename(filename).lower()
    if "catalog" in name or "product" in name:
        return "catalog"
    if "faq" in name:
        return "faq"
    if any(k in name for k in ("policy", "return", "shipping", "terms")):
        return "policy"
    if sum(1 for l in lines if PRODUCT_RE.match(l)) >= 2:
        return "catalog"
    if sum(1 for l in lines if FAQ_Q_RE.match(l)) >= 2:
        return "faq"
    if sum(1 for l in lines if _is_policy_head(l)) >= 2:
        return "policy"
    if sum(1 for l in lines if _is_caps_head(l)) >= 2:
        return "sections"
    return "generic"


def _make_chunk(doc, section, body, dtype):
    header = f"[{doc} > {section}]" if section else f"[{doc}]"
    return {
        "text": f"{header}\n{body.strip()}",
        "metadata": {"doc": doc, "section": section, "type": dtype},
    }


def _split_long(body, max_chars):
    if len(body) <= max_chars:
        return [body]
    parts, cur = [], ""
    for line in body.splitlines():
        while len(line) > max_chars:
            cut = line.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:cut])
            line = line[cut:].lstrip()
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def _emit(doc, section, body, dtype, out, max_chars):
    body = body.strip()
    if not body:
        return
    parts = _split_long(body, max_chars)
    for i, p in enumerate(parts, 1):
        sec = section if len(parts) == 1 else f"{section} (part {i}/{len(parts)})"
        out.append(_make_chunk(doc, sec, p, dtype))


def _split_by_markers(lines, is_start):
    sections, title, buf = [], None, []
    for line in lines:
        t = is_start(line)
        if t is not None:
            if title is not None or any(l.strip() for l in buf):
                sections.append((title, buf))
            title, buf = t, []
        else:
            buf.append(line)
    if title is not None or any(l.strip() for l in buf):
        sections.append((title, buf))
    return sections


def chunk_document(filename, text, max_chars=MAX_CHARS):
    text = text.replace("\r\n", "\n")
    # "(Demo store. All ... fictional.)" নোটটা বাদ; কাজের তথ্য (যেমন inches) থাকে
    text = re.sub(r"Demo store\.\s*(All [^.]*?fictional\.)?\s*", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    # প্রথম লাইন = ডকুমেন্টের শিরোনাম -> ডকুমেন্টের নাম
    title_line = lines[0] if lines else ""
    if title_line.strip() and _is_caps_head(title_line):
        lines.pop(0)
        doc = _nice(title_line.split(" - ", 1)[-1])
    else:
        doc = _nice(re.sub(r"^\d+[-_ ]*|[_\-]+", " ", os.path.splitext(os.path.basename(filename))[0]))
    dtype = _doc_type(filename, lines)
    out = []

    if dtype == "catalog":
        def start(line):
            m = PRODUCT_RE.match(line)
            if m:
                return m.group("title").strip()
            return _nice(line) if _is_caps_head(line) else None
        for title, buf in _split_by_markers(lines, start):
            _emit(doc, title or "Overview", "\n".join(buf), dtype, out, max_chars)

    elif dtype == "faq":
        category, cur_q, buf = None, None, []

        def flush():
            if cur_q is None:
                return
            sec = cur_q[:80] if not category else f"{category} > {cur_q[:80]}"
            _emit(doc, sec, f"Q: {cur_q}\n" + "\n".join(buf).strip(), dtype, out, max_chars)

        for line in lines:
            m = FAQ_Q_RE.match(line)
            if m:
                flush()
                cur_q, buf = m.group("q").strip(), []
            elif _is_caps_head(line):
                flush()
                cur_q, buf, category = None, [], _nice(line)
            elif cur_q is not None:
                buf.append(line)
        flush()

    elif dtype == "policy":
        def start(line):
            if _is_policy_head(line):
                m = POLICY_HEAD_RE.match(line)
                return f"{m.group('num')}. {m.group('title').strip()}"
            return None
        for title, buf in _split_by_markers(lines, start):
            _emit(doc, title or "Overview", "\n".join(buf), dtype, out, max_chars)

    elif dtype == "sections":
        def start(line):
            return _nice(line) if _is_caps_head(line) else None
        for title, buf in _split_by_markers(lines, start):
            _emit(doc, title or "Overview", "\n".join(buf), dtype, out, max_chars)

    else:
        paras = [p for p in re.split(r"\n\s*\n", "\n".join(lines)) if p.strip()]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) + 2 > max_chars:
                _emit(doc, "", cur, dtype, out, max_chars)
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        _emit(doc, "", cur, dtype, out, max_chars)

    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "docs"
    paths = (
        [os.path.join(target, f) for f in sorted(os.listdir(target))]
        if os.path.isdir(target)
        else [target]
    )
    for path in paths:
        if not path.lower().endswith((".md", ".txt")):
            continue
        with open(path, encoding="utf-8") as f:
            chunks = chunk_document(path, f.read())
        print(f"\n===== {path}: {len(chunks)} chunks =====")
        for c in chunks:
            print("-" * 40)
            print(c["text"])
