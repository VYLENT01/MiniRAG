"""
Strict Extractive Prompt with Multi-Chunk Fusion.
"""

from typing import List
from minirag.prompts.base import BasePromptBuilder
from minirag.models.search import SearchResult


class QACitationPromptBuilder(BasePromptBuilder):
    def build(self, context: List[SearchResult], question: str) -> str:
        context_str = ""
        for i, result in enumerate(context, start=1):
            context_str += f"[{i}]:\n{result.text}\n\n"

        prompt = f"""You are a strict "Verbatim Extraction API". 
You are NOT allowed to generate new text. You are NOT allowed to paraphrase.
Your ONLY job is to find exact phrases in the CONTEXT that answer the QUESTION.

RULES:
1. Read the CONTEXT carefully.
2. Find exact sentences/phrases from the text that answer the question.
3. Copy them EXACTLY as they appear. Do not change a single word. Do not add your own explanations.
4. MULTI-CHUNK FUSION: If the answer is split across multiple sentences or chunks (e.g., a list of 4 stages), combine those exact phrases into a single string using commas, but DO NOT add connecting words like "and" or "then". Just stitch the exact quotes together.
5. Output ONLY a valid JSON object with a single key "quotes" which is a list of strings.
6. Language of quotes MUST match the language of the question.

CONTEXT:
{context_str}

QUESTION: {question}

JSON OUTPUT:"""
        return prompt