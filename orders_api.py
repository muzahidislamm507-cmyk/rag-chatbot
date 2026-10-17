"""
UrbanWear ডেমো স্টোরের অর্ডার ও প্রোডাক্ট সার্ভারে রাখার মডিউল।

স্টোর, অ্যাডমিন প্যানেল আর চ্যাটবট তিনজনই এখান থেকে একই তথ্য পড়বে।
ডেটা `store_data.json` ফাইলে সেভ হয় (ফাইল না থাকলে ডেমো ডেটা দিয়ে নিজে তৈরি হয়)।

এন্ডপয়েন্ট:
    GET   /api/products                  : সবার জন্য, প্রোডাক্ট ও স্টক
    POST  /api/orders                    : সবার জন্য, নতুন অর্ডার (দাম সার্ভারই হিসাব করে)
    GET   /api/orders/{id}               : সবার জন্য, ট্র্যাকিং (নাম/ফোন/ঠিকানা দেখায় না)
    GET   /api/admin/orders              : শুধু ADMIN_KEY দিয়ে, সব অর্ডার
    PATCH /api/admin/orders/{id}         : শুধু ADMIN_KEY দিয়ে, স্ট্যাটাস বদল
    PATCH /api/admin/products/{id}       : শুধু ADMIN_KEY দিয়ে, দাম ও স্টক বদল

চ্যাটবটের জন্য: order_reply(query) ফাংশন প্রশ্নে অর্ডার নম্বর পেলে বাংলায় উত্তর বানিয়ে দেয়।
"""

import os
import re
import json
import time
import hmac
import logging
import threading
from collections import defaultdict, deque
from datetime import date

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

log = logging.getLogger("orders_api")

DATA_FILE = os.path.join(os.getenv("DATA_DIR", "."), "store_data.json")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
STATUS_BN = [
    "অর্ডার গ্রহণ করা হয়েছে",
    "প্যাকিং হয়ে গেছে, কুরিয়ারের জন্য অপেক্ষা করছে",
    "কুরিয়ারে পাঠানো হয়েছে",
    "ডেলিভারি সম্পন্ন হয়েছে",
]
_lock = threading.Lock()


# ---------------------------------------------------------
# রেট লিমিট (প্রতি IP, প্রতি মিনিটে ৩০টা) ও অ্যাডমিন চাবি যাচাই
# ---------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 30:
        raise HTTPException(status_code=429, detail="অনেক বেশি রিকোয়েস্ট হয়ে গেছে, একটু পর আবার চেষ্টা করুন।")
    q.append(now)


def require_admin(request: Request, x_admin_key: str = Header(default="")):
    if not ADMIN_KEY:
        raise HTTPException(status_code=503, detail="সার্ভারে ADMIN_KEY সেট করা নেই, তাই এই কাজ বন্ধ আছে।")
    if not hmac.compare_digest(x_admin_key.encode("utf-8"), ADMIN_KEY.encode("utf-8")):
        raise HTTPException(status_code=401, detail="অনুমতি নেই, সঠিক admin key দিন।")


router = APIRouter(prefix="/api", dependencies=[Depends(rate_limit)])


# ---------------------------------------------------------
# ডেটা ফাইল পড়া ও লেখা
# ---------------------------------------------------------
def _default_data() -> dict:
    products = [
        (1, "Daily Tee", "T-shirts", 650, 25, ["S", "M", "L", "XL"]),
        (2, "Heavy Tee", "T-shirts", 850, 18, ["S", "M", "L", "XL"]),
        (3, "Commuter Hoodie", "Hoodies", 1850, 12, ["M", "L", "XL"]),
        (4, "Zip Hoodie", "Hoodies", 2100, 4, ["M", "L", "XL"]),
        (5, "Stretch Jogger", "Bottoms", 1450, 20, ["S", "M", "L", "XL"]),
        (6, "Cargo Pant", "Bottoms", 1750, 9, ["30", "32", "34", "36"]),
        (7, "Canvas Cap", "Accessories", 480, 30, ["One size"]),
        (8, "Tote Bag", "Accessories", 550, 2, ["One size"]),
    ]
    return {
        "products": [
            {"id": i, "name": n, "cat": c, "price": p, "stock": s, "sizes": z}
            for i, n, c, p, s, z in products
        ],
        "orders": {
            "UW-1001": {
                "stage": 2, "name": "Rahim Uddin", "phone": "01711-000001", "addr": "Mirpur, Dhaka",
                "items": [{"name": "Commuter Hoodie", "size": "L", "qty": 1, "price": 1850}],
                "total": 1850, "date": "2026-10-14",
            },
            "UW-1002": {
                "stage": 1, "name": "Nusrat Jahan", "phone": "01811-000002", "addr": "Khulna Sadar",
                "items": [{"name": "Daily Tee", "size": "M", "qty": 2, "price": 650}],
                "total": 1300, "date": "2026-10-15",
            },
        },
    }


def _save(data: dict):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, DATA_FILE)  # মাঝপথে বন্ধ হয়ে গেলেও ফাইল নষ্ট হয় না


def _load() -> dict:
    if not os.path.exists(DATA_FILE):
        data = _default_data()
        _save(data)
        return data
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# রিকোয়েস্টের ছাঁচ (ভুল বা অতিরিক্ত বড় ইনপুট এখানেই আটকে যায়)
# ---------------------------------------------------------
class OrderItem(BaseModel):
    product_id: int
    size: str = Field(..., min_length=1, max_length=20)
    qty: int = Field(..., ge=1, le=20)


class OrderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=3, max_length=20)
    addr: str = Field(..., min_length=3, max_length=300)
    items: list[OrderItem] = Field(..., min_length=1, max_length=20)


class StagePatch(BaseModel):
    stage: int = Field(..., ge=0, le=3)


class ProductPatch(BaseModel):
    price: int | None = Field(default=None, ge=0, le=10_000_000)
    stock: int | None = Field(default=None, ge=0, le=1_000_000)


# ---------------------------------------------------------
# সবার জন্য খোলা এন্ডপয়েন্ট
# ---------------------------------------------------------
@router.get("/products")
def list_products():
    with _lock:
        return {"products": _load()["products"]}


@router.post("/orders")
def create_order(req: OrderCreate):
    with _lock:
        data = _load()
        prods = {p["id"]: p for p in data["products"]}

        need: dict[int, int] = defaultdict(int)
        for it in req.items:
            p = prods.get(it.product_id)
            if not p:
                raise HTTPException(status_code=400, detail="একটা প্রোডাক্ট খুঁজে পাওয়া যায়নি।")
            if it.size not in p["sizes"]:
                raise HTTPException(status_code=400, detail=f"'{p['name']}'-এর এই সাইজ নেই।")
            need[it.product_id] += it.qty
        for pid, q in need.items():
            if prods[pid]["stock"] < q:
                raise HTTPException(status_code=409, detail=f"'{prods[pid]['name']}'-এর যথেষ্ট স্টক নেই।")

        items, total = [], 0
        for it in req.items:
            p = prods[it.product_id]
            items.append({"name": p["name"], "size": it.size, "qty": it.qty, "price": p["price"]})
            total += p["price"] * it.qty
        for pid, q in need.items():
            prods[pid]["stock"] -= q

        nums = [int(k.split("-")[1]) for k in data["orders"]]
        order_id = f"UW-{max(nums + [1002]) + 1}"
        data["orders"][order_id] = {
            "stage": 0, "name": req.name.strip(), "phone": req.phone.strip(), "addr": req.addr.strip(),
            "items": items, "total": total, "date": date.today().isoformat(),
        }
        _save(data)
    log.info("নতুন অর্ডার: %s (মোট %s)", order_id, total)
    return {"order_id": order_id, "total": total}


@router.get("/orders/{order_id}")
def track_order(order_id: str):
    with _lock:
        o = _load()["orders"].get(order_id.upper())
    if not o:
        raise HTTPException(status_code=404, detail="এই নম্বরে কোনো অর্ডার পাওয়া যায়নি।")
    # ট্র্যাকিং পাবলিক, তাই নাম, ফোন আর ঠিকানা ইচ্ছা করেই বাদ রাখা হয়েছে
    return {
        "order_id": order_id.upper(), "stage": o["stage"], "status": STATUS_BN[o["stage"]],
        "items": [{"name": i["name"], "size": i["size"], "qty": i["qty"]} for i in o["items"]],
        "date": o["date"],
    }


# ---------------------------------------------------------
# শুধু অ্যাডমিনের জন্য
# ---------------------------------------------------------
@router.get("/admin/orders", dependencies=[Depends(require_admin)])
def admin_orders():
    with _lock:
        orders = _load()["orders"]
    rows = [{"order_id": k, **v} for k, v in orders.items()]
    rows.sort(key=lambda r: int(r["order_id"].split("-")[1]), reverse=True)
    return {"orders": rows}


@router.patch("/admin/orders/{order_id}", dependencies=[Depends(require_admin)])
def admin_set_stage(order_id: str, req: StagePatch):
    with _lock:
        data = _load()
        o = data["orders"].get(order_id.upper())
        if not o:
            raise HTTPException(status_code=404, detail="এই নম্বরে কোনো অর্ডার নেই।")
        o["stage"] = req.stage
        _save(data)
    return {"order_id": order_id.upper(), "stage": req.stage}


@router.patch("/admin/products/{product_id}", dependencies=[Depends(require_admin)])
def admin_edit_product(product_id: int, req: ProductPatch):
    with _lock:
        data = _load()
        p = next((x for x in data["products"] if x["id"] == product_id), None)
        if not p:
            raise HTTPException(status_code=404, detail="প্রোডাক্ট খুঁজে পাওয়া যায়নি।")
        if req.price is not None:
            p["price"] = req.price
        if req.stock is not None:
            p["stock"] = req.stock
        _save(data)
    return p


# ---------------------------------------------------------
# চ্যাটবটের জন্য: প্রশ্নে অর্ডার নম্বর থাকলে আসল অর্ডার দেখে বাংলায় উত্তর
# কিছু না মিললে None ফেরত দেয়, তখন বট সাধারণ নিয়মে ডকুমেন্ট থেকে উত্তর দেবে।
# ---------------------------------------------------------
_ORDER_ID = re.compile(r"(?<![A-Za-z0-9])UW[-\s]?(\d{3,6})(?!\d)", re.I)
_ORDER_INTENT = re.compile(
    r"(অর্ডার|order|পার্সেল|parcel).*(কোথায়|কই|স্ট্যাটাস|ট্র্যাক|track|status|where)"
    r"|(কোথায়|কই|স্ট্যাটাস|ট্র্যাক|track|status|where).*(অর্ডার|order|পার্সেল|parcel)",
    re.I | re.S,
)


def order_reply(query: str) -> str | None:
    m = _ORDER_ID.search(query)
    if not m:
        if _ORDER_INTENT.search(query):
            return "অর্ডারের অবস্থা দেখতে আপনার অর্ডার নম্বরটা লিখুন (যেমন: UW-1001)।"
        return None

    order_id = f"UW-{m.group(1)}"
    with _lock:
        o = _load()["orders"].get(order_id)
    if not o:
        return f"দুঃখিত, {order_id} নম্বরে কোনো অর্ডার খুঁজে পাইনি। নম্বরটা আরেকবার দেখে নিন।"

    items = ", ".join(f"{i['name']} ({i['size']}) x{i['qty']}" for i in o["items"])
    return f"আপনার অর্ডার {order_id}-এর অবস্থা: {STATUS_BN[o['stage']]}।\nপণ্য: {items}\nমোট: ৳{o['total']}"
