"""
api.py-তে অর্ডার এপিআই যোগ করার এককালীন স্ক্রিপ্ট।

চালানোর নিয়ম (api.py যে ফোল্ডারে আছে সেখানে):
    python patch_api.py

এটা যা করে:
    1. api.py-এর ব্যাকআপ রাখে (api.py.bak)
    2. FastAPI অ্যাপ বানানোর লাইনের ঠিক পরে দুটো লাইন যোগ করে
    3. আবার চালালে কিছু বদলায় না (আগে থেকে যোগ করা থাকলে জানিয়ে দেয়)
"""

import re
import shutil
import sys

FILE = "api.py"
MARKER = "orders_router"
NEW_LINES = (
    "from orders_api import router as orders_router  # অর্ডার ও প্রোডাক্ট এপিআই\n"
    "app.include_router(orders_router)\n"
)

try:
    with open(FILE, encoding="utf-8") as f:
        src = f.read()
except FileNotFoundError:
    sys.exit("api.py পাওয়া যায়নি। এই স্ক্রিপ্টটা api.py-এর পাশের ফোল্ডারে রেখে চালান।")

if MARKER in src:
    sys.exit("api.py-তে আগেই যোগ করা আছে, নতুন করে কিছু বদলানো হয়নি।")

pattern = re.compile(r"^app = FastAPI\(.*\)[ \t]*\r?\n", re.M)
matches = pattern.findall(src)
if len(matches) != 1:
    sys.exit("'app = FastAPI(...)' লাইনটা ঠিক একবার পাওয়া যায়নি, তাই কিছু বদলানো হয়নি। আমাকে জানান।")

shutil.copyfile(FILE, FILE + ".bak")
patched = pattern.sub(lambda m: m.group(0) + NEW_LINES, src, count=1)
with open(FILE, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

print("হয়ে গেছে। api.py আপডেট হয়েছে, আগের কপি api.py.bak নামে রাখা আছে।")
