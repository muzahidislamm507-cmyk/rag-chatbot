"""
একটা ছোট টেস্ট সেট দিয়ে RAG বট যাচাই করার স্ক্রিপ্ট (#৭)।

প্রতিটা টেস্ট কেসে:
    - question: প্রশ্ন
    - expected_keywords: উত্তরে যেসব শব্দ/তথ্য থাকা উচিত

চেক করে:
    1. Retrieval hit  — প্রাসঙ্গিক চাঙ্ক আদৌ খুঁজে পেয়েছে কিনা
    2. Keyword recall — উত্তরে expected keyword-গুলো আছে কিনা
    3. Groundedness   — উত্তরের কথাগুলো retrieved context-এর সাথে
                        (মোটামুটি) মিলছে কিনা, নাকি সম্পূর্ণ বাইরের
                        কিছু বলে দিয়েছে (হ্যালুসিনেশনের সহজ সিগন্যাল)

এটা কোনো fancy eval framework না — কিন্তু প্রতিবার কোড বদলানোর পর
`python eval.py` চালালেই বোঝা যাবে কিছু ভেঙেছে কিনা।
"""

from rag_core import get_or_build_index, answer_query

TEST_CASES = [
    # --- প্রোডাক্ট ক্যাটালগ (এক প্রোডাক্ট = এক চাঙ্ক, তাই রং/দাম ঠিক প্রোডাক্টের সাথে আসা উচিত) ---
    {
        "question": "Oversized Hoodie কী কী রঙে পাওয়া যায়?",
        "expected_keywords": ["Cream", "Dusty Pink"],
    },
    {
        "question": "Slim Fit Jeans-এর দাম কত?",
        "expected_keywords": ["64.99", "৬৪.৯৯"],
    },
    # --- পলিসি ---
    {
        "question": "রিটার্ন করার সময়সীমা কত দিন?",
        "expected_keywords": ["30", "৩০"],
    },
    {
        "question": "US-এ standard shipping-এ কত দিন লাগে আর খরচ কত?",
        "expected_keywords": ["3-5", "৩-৫", "4.99", "৪.৯৯"],
    },
    # --- Final Sale (ক্যাটালগ + পলিসি + CURRENT SALE চাঙ্ক মিলিয়ে) ---
    {
        "question": "মোজা কি ফেরত দেওয়া যায়?",
        "expected_keywords": ["Final Sale", "ফাইনাল সেল", "যাবে না", "যায় না"],
    },
    {
        "question": "Navy রঙের Puffer Vest কি ফেরত দেওয়া যাবে?",
        "expected_keywords": ["Final Sale", "ফাইনাল সেল", "যাবে না", "যায় না"],
    },
    # --- FAQ ---
    {
        "question": "WELCOME10 কোড দিয়ে কত ছাড় পাওয়া যায়?",
        "expected_keywords": ["10%", "১০%", "10 শতাংশ", "১০ শতাংশ"],
    },
    # ডকুমেন্টে নেই এমন প্রশ্ন — মডেল যেন হ্যালুসিনেট না করে সেটা চেক করতে
    {
        "question": "UrbanWear-এর CEO কে?",
        "expected_keywords": ["পাওয়া যায়নি"],
    },
]


def word_overlap_ratio(answer: str, context_texts: list[str]) -> float:
    """উত্তরের কতগুলো (৩+ অক্ষরের) শব্দ context-এ আছে — খুবই সাধারণ groundedness signal।"""
    context_words = set()
    for t in context_texts:
        context_words.update(w for w in t.split() if len(w) > 2)

    answer_words = [w for w in answer.split() if len(w) > 2]
    if not answer_words:
        return 1.0
    matched = sum(1 for w in answer_words if w in context_words)
    return matched / len(answer_words)


def run_eval():
    index, chunks = get_or_build_index()

    total = len(TEST_CASES)
    keyword_hits = 0
    retrieval_hits = 0
    low_groundedness = []

    for i, case in enumerate(TEST_CASES, 1):
        result = answer_query(case["question"], index, chunks)
        answer = result["answer"]
        retrieved_texts = [r["text"] for r in result["retrieved"]]

        found_kw = [kw for kw in case["expected_keywords"] if kw in answer]
        kw_ok = len(found_kw) > 0
        keyword_hits += int(kw_ok)
        retrieval_hits += int(len(retrieved_texts) > 0)

        groundedness = word_overlap_ratio(answer, retrieved_texts)
        if groundedness < 0.3 and "পাওয়া যায়নি" not in answer:
            low_groundedness.append(case["question"])

        print(f"[{i}/{total}] {case['question']}")
        print(f"   উত্তর            : {answer[:150]}")
        print(f"   keyword match    : {'✅' if kw_ok else '❌'} ({found_kw or 'কিছু মেলেনি'})")
        print(f"   groundedness     : {groundedness:.2f}")
        print()

    print("=" * 50)
    print(f"Retrieval hit rate : {retrieval_hits}/{total}")
    print(f"Keyword recall     : {keyword_hits}/{total}")
    if low_groundedness:
        print(f"⚠️ কম groundedness (সম্ভাব্য হ্যালুসিনেশন): {low_groundedness}")
    else:
        print("✅ সব উত্তরই context-এর সাথে মোটামুটি সঙ্গতিপূর্ণ মনে হচ্ছে।")


if __name__ == "__main__":
    run_eval()
