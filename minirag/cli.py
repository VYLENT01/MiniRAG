"""
MiniRAG CLI.
"""
import sys
from minirag import MiniRAG, MiniRAGError

def print_separator():
    print("-" * 60)

def main():
    print("Initializing MiniRAG...")
    rag = MiniRAG()
    debug_mode = False

    while True:
        mode_str = " (DEBUG ON)" if debug_mode else ""
        print(f"\n=== MiniRAG Main Menu{mode_str} ===")
        print("1. Index Document")
        print("2. Ask Question")
        print("3. List Documents")
        print("4. Delete Document")
        print("5. Rebuild Index")
        print("6. Statistics")
        print("8. Toggle Debug Mode")
        print("0. Exit")
        
        choice = input("Select an option: ").strip()

        if choice == "0":
            sys.exit(0)
            
        elif choice == "8":
            debug_mode = not debug_mode
            print(f"\nDebug mode turned {'ON' if debug_mode else 'OFF'}.")
            continue
            
        elif choice == "1":
            path = input("Enter absolute document path: ").strip().strip('"').strip("'") 
            try:
                meta = rag.add_document(path)
                print(f"\n[SUCCESS] Indexed: {meta.file_name} | Chunks: {meta.chunk_count}")
            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")
                
        elif choice == "2":
            question = input("Enter your question: ").strip()
            if not question: continue
            try:
                answer = rag.ask(question)
                
                print_separator()
                if answer.confidence_level == "REJECTED":
                    print("[SYSTEM WARNING] LLM output failed Grounding Check (Hallucination blocked).")
                
                print(answer.text)
                print_separator()

                if debug_mode:
                    conf_color = "\033[92m" if answer.confidence_level == "HIGH" else ("\033[93m" if answer.confidence_level == "MEDIUM" else "\033[91m")
                    print(f"\n[CONFIDENCE: {conf_color}{answer.confidence_level}\033[0m | Score: {answer.confidence_score:.2f}]")
                    
                    if answer.citations:
                        print("\n--- DEBUG: Retrieved Chunks ---")
                        for i, cit in enumerate(answer.citations, 1):
                            page_info = f"Page {cit.page}" if cit.page else "No Page"
                            print(f"[{i}] {cit.document_name} | {page_info} | Chunk: {cit.chunk_id} | Sim: {cit.similarity_score:.2f}")
                            print(f"    >> {cit.exact_snippet}\n")
                    else:
                        print("\n[DEBUG] No chunks passed threshold.")
                        
            except MiniRAGError as e:
                print(f"\n[ERROR] {e}")
                
        elif choice == "3":
            docs = rag.list_documents()
            if not docs: print("\nNo documents indexed yet.")
            else:
                for doc in docs: print(f" - [{doc.uuid}] {doc.file_name}")
                
        elif choice == "4":
            doc_id = input("Enter Document UUID to delete: ").strip()
            try:
                if rag.delete_document(doc_id): print(f"\n[SUCCESS] Deleted. Run 'Rebuild' to clean vector store.")
            except MiniRAGError as e: print(f"\n[ERROR] {e}")
            
        elif choice == "5":
            print("\nRebuilding index... (This fixes Page Numbers)")
            try:
                count = rag.rebuild()
                print(f"\n[SUCCESS] Rebuilt {count} documents.")
            except MiniRAGError as e: print(f"\n[ERROR] {e}")
            
        elif choice == "6":
            docs = rag.list_documents()
            print(f"\nDocs: {len(docs)} | Embedder: {rag.config.embedding_provider} | LLM: {rag.config.primary_llm_provider}")

        else:
            print("\nInvalid option.")

if __name__ == "__main__":
    main()