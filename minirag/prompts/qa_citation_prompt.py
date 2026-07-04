"""
QA with Citations Prompt Implementation.
"""

from typing import List
from minirag.prompts.base import BasePromptBuilder
from minirag.models.search import SearchResult


class QACitationPromptBuilder(BasePromptBuilder):
    """Builds a prompt optimized for strict, dry data extraction."""

    def build(self, context: List[SearchResult], question: str) -> str:
        if not context:
            return f"QUESTION:\n{question}\n\nNo relevant context was found. Reply ONLY with: پاسخ این سؤال در اسناد موجود یافت نشد."

        context_str = ""
        for i, result in enumerate(context, start=1):
            page_info = result.page if result.page else "Unknown"
            doc_name = result.document_name if result.document_name else "Unknown Document"
            
            context_str += f"[{i}] ({doc_name} - {page_info}):\n{result.text}\n\n"

        # پرامپت سیستم برای خاموش کردن رفتارهای ذاتی LLM
        prompt = f"""SYSTEM BEHAVIOR OVERRIDE: You are a "Data Extraction Engine". You do NOT think out loud. You do NOT act like a chatbot.

BANNED PHRASES (Using any of these will cause a system failure):
- "According to", "Based on", "Translation:", "Please note"
- "Wait", "I made a mistake", "So, the answer is"
- Any conversational filler, introductions, or conclusions.

TASK: Extract the answer to the QUESTION from the CONTEXT. Output ONLY the extracted text.

STRICT RULES:
1. LANGUAGE: Match the question's language 100%. If Persian, output Persian. NO English words.
2. SEMANTIC EXTRACTION (CRITICAL): Do not act dumb. If the question asks about "A" and the text discusses "B" (but B means the same thing as A, e.g., "تنوع و تجدد" vs "دلزدگی و آرزوهای متعدد"), extract the text about B. Do NOT say "not found" if a semantic link exists.
3. COMPLETENESS: If asked for a list, extract ALL items.
4. CITATIONS: Append [Source X] at the end of EVERY extracted sentence.
5. NOT FOUND: ONLY output exactly "پاسخ این سؤال در اسناد موجود یافت نشد." if the text is 100% completely off-topic.

CONTEXT:
{context_str}

QUESTION:
{question}

EXTRACTION:"""

        return prompt