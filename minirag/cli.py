"""
MiniRAG Command Line Interface.
"""

import sys
from minirag import MiniRAG, MiniRAGError, Config


def print_separator():
    print("-" * 60)

def main():
    print("Initializing MiniRAG...")
    rag = MiniRAG()

    while True:
        print("\n=== MiniRAG Main Menu ===")
        print("1. Index Document")
        print("2. Ask Question")
        print("3. List Documents")
        print("4. Delete Document")
        print("5. Rebuild Index")
        print("6. Statistics")
        print("0. Exit")
        
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting MiniRAG. Goodbye!")
            sys.exit(0)

        elif choice == "1":
            path = input("Enter absolute document path: ").strip()
            path = path.strip('"').strip("'") 
            try:
                meta = rag.add_document(path)
                print(f"\n[SUCCESS] Indexed: {meta.file_name}")
                print(f"UUID: {meta.uuid}")
                print(f"Chunks created: {meta.chunk_count}")
            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")

        elif choice == "2":
            question = input("Enter your question: ").strip()
            if not question:
                continue
            try:
                print("\nExtracting information...")
                answer = rag.ask(question)
                
                print_separator()
                # نمایش سطح اطمینان
                conf_color = "\033[92m" if answer.confidence_level == "HIGH" else ("\033[93m" if answer.confidence_level == "MEDIUM" else "\033[91m")
                print(f"[Confidence: {conf_color}{answer.confidence_level}\033[0m | Score: {answer.confidence_score:.2f}]")
                print_separator()
                print(answer.text)
                print_separator()
                
                # غنی‌سازی بخش نمایش منابع با UUID کامل
                if answer.citations:
                    print("\n--- Referenced Sources ---")
                    for i, cit in enumerate(answer.citations, 1):
                        page_info = f"Page {cit.page}" if cit.page else "Unknown Page"
                        print(f"[{i}] {cit.document_name} | {page_info}")
                        print(f"    Chunk ID: {cit.chunk_id} | Similarity: {cit.similarity_score:.2f}")
                        print(f"    > {cit.exact_snippet[:120]}...\n")
                else:
                    if not answer.is_faithful:
                        print("\n[SYSTEM] Blocked by Similarity Threshold (No relevant chunks found).")

            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")

        elif choice == "3":
            docs = rag.list_documents()
            if not docs:
                print("\nNo documents indexed yet.")
            else:
                print(f"\nTotal Documents: {len(docs)}")
                for doc in docs:
                    print(f" - [{doc.uuid}] {doc.file_name} ({doc.file_type})")

        elif choice == "4":
            doc_id = input("Enter Document UUID to delete: ").strip()
            try:
                if rag.delete_document(doc_id):
                    print(f"\n[SUCCESS] Document {doc_id} deleted.")
                    print("Note: Run 'Rebuild Index' to clean up the vector store.")
            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")

        elif choice == "5":
            print("\nRebuilding vector index... This may take a moment.")
            try:
                count = rag.rebuild()
                print(f"\n[SUCCESS] Rebuilt index for {count} documents.")
            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")

        elif choice == "6":
            docs = rag.list_documents()
            total_chunks = sum(d.chunk_count for d in docs)
            print("\n--- Statistics ---")
            print(f"Total Documents: {len(docs)}")
            print(f"Total Chunks: {total_chunks}")
            print(f"Embedding Model: {rag.config.embedding_provider}")
            print(f"LLM Provider: {rag.config.primary_llm_provider}")
            print(f"Similarity Threshold: {rag.config.similarity_threshold}")
            print(f"Data Directory: {rag.config.base_data_dir}")

        else:
            print("\nInvalid option. Please try again.")


if __name__ == "__main__":
    main()