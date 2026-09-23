"""
টার্মিনাল (CLI) থেকে RAG বট চালানোর স্ক্রিপ্ট।
সব মূল লজিক rag_core.py-তে; এই ফাইল শুধু ইউজার ইন্টারঅ্যাকশন সামলায়।
"""

from rag_core import (
    logger,
    get_or_build_index,
    add_document_to_store,
    save_vector_store,
    create_chat_session,
    stream_answer_query,
)


def main():
    try:
        index, chunks = get_or_build_index()
    except Exception as e:
        logger.critical(f"ইনডেক্স বানাতে ব্যর্থ হয়েছি, প্রোগ্রাম বন্ধ করা হচ্ছে: {e}")
        print("❌ শুরুর ডকুমেন্ট থেকে ইনডেক্স বানানো যায়নি। docs/ ফোল্ডারে .txt/.pdf ফাইল আছে কিনা দেখুন, বা rag_pipeline.log চেক করুন।")
        return

    session = create_chat_session()  # #Conversation Memory: পুরো সেশন জুড়ে আগের প্রশ্ন-উত্তর মনে থাকবে

    print("\n✅ RAG বট রেডি! প্রশ্ন করুন (আগের প্রশ্নের প্রসঙ্গও মনে রাখবে)")
    print("   (নতুন ডকুমেন্ট যোগ করতে: 'add <file_path>', বের হতে: exit)\n")

    while True:
        try:
            query = input("প্রশ্ন: ")
        except (EOFError, KeyboardInterrupt):
            logger.info("ইউজার প্রোগ্রাম বন্ধ করে দিয়েছেন।")
            break

        if query.strip().lower() in ("exit", "quit", "বন্ধ"):
            break
        if not query.strip():
            continue

        if query.strip().lower().startswith("add "):
            new_file = query.strip()[4:].strip()
            index, chunks = add_document_to_store(new_file, index, chunks)
            save_vector_store(index, chunks)
            continue

        try:
            print("\nউত্তর: ", end="", flush=True)
            sources, retrieved = [], []
            for event in stream_answer_query(query, index, chunks, session=session):
                if event["type"] == "chunk":
                    print(event["text"], end="", flush=True)  # #Streaming Answer: টুকরো টুকরো প্রিন্ট
                elif event["type"] == "error":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "done":
                    sources = event["sources"]
                    retrieved = event["retrieved"]

            print()  # নতুন লাইনে যাওয়া
            if sources:
                print(f"📚 সূত্র: {', '.join(sources)}")
            print()

            logger.info("রিট্রিভ হওয়া চাঙ্কগুলো (ডিবাগ):")
            for i, c in enumerate(retrieved, 1):
                logger.info(f"  [{i}] ({c['source']}) {c['text']}...")
        except Exception as e:
            logger.exception(f"প্রশ্ন প্রসেস করতে অপ্রত্যাশিত সমস্যা হয়েছে: {e}")
            print("\n❌ দুঃখিত, এই প্রশ্নে একটা সমস্যা হয়েছে। rag_pipeline.log ফাইলে বিস্তারিত দেখুন।\n")


if __name__ == "__main__":
    main()
