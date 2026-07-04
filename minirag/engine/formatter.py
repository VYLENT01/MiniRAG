"""
Deterministic Answer Formatter with Persian Grounding Check.
"""

import json
import re
from typing import List

from minirag.models.answer import Answer, Citation
from minirag.models.search import SearchResult


class AnswerFormatter:
    """Assembles answers strictly from verified verbatim quotes."""

    @staticmethod
    def normalize_persian(text: str) -> str:
        """
        Standardizes Persian/Arabic text for reliable substring matching.
        Strips diacritics, normalizes ZWNJ, characters, and removes punctuation.
        """
        # 1. Remove Arabic Diacritics (Fatha, Kasra, Damma, Tanwin, Shadda, etc.)
        diacritics_re = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
        text = diacritics_re.sub('', text)
        
        # 2. Replace Zero-Width Non-Joiner (ZWNJ) with space (CRITICAL FOR PERSIAN)
        # This makes "خودآگاهی" and "خود آگاهی" identical to the checker.
        text = text.replace('\u200c', ' ')
        
        # 3. Normalize Persian/Arabic character variations
        text = text.replace('ی', 'ي').replace('ک', 'ك').replace('ة', 'ه')
        
        # 4. Remove all punctuation and special characters (keep only words and spaces)
        text = re.sub(r'[^\w\s]', '', text)
        
        # 5. Normalize multiple spaces to a single space and lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        return text

    @classmethod
    def format(cls, llm_raw_output: str, retrieved_chunks: List[SearchResult], threshold: float) -> Answer:
        quotes = cls._extract_json_list(llm_raw_output, "quotes")
        
        # Build a normalized pool of all text from retrieved chunks
        chunk_texts_combined = " ".join([c.text for c in retrieved_chunks])
        clean_corpus = cls.normalize_persian(chunk_texts_combined)

        verified_quotes = []
        for quote in quotes:
            clean_quote = cls.normalize_persian(quote)
            # GROUNDING CHECK: Is the normalized quote inside the normalized corpus?
            if clean_quote and len(clean_quote) > 10 and clean_quote in clean_corpus:
                verified_quotes.append(quote) # Return the ORIGINAL quote, not the normalized one

        if not retrieved_chunks:
            return Answer(text="پاسخ این سؤال در اسناد موجود یافت نشد.", is_faithful=False, confidence_score=0.0, confidence_level="NONE", citations=[])

        # Calculate Margin-based Confidence
        scores = sorted([c.similarity_score for c in retrieved_chunks], reverse=True)
        if len(scores) >= 2:
            margin = scores[0] - scores[1]
            confidence_score = scores[0] - (margin * 0.2)
        else:
            confidence_score = scores[0]
            
        confidence_level = "HIGH" if confidence_score >= 0.6 else ("MEDIUM" if confidence_score >= 0.4 else "LOW")

        if not verified_quotes:
            text = "سیستم نتوانست پاسخ دقیقی از متن استخراج کند."
            confidence_level = "REJECTED"
            confidence_score = 0.0
        else:
            # Pure Extractive Assembly
            text = "\n".join(f"• {q}" for q in verified_quotes)

        citations = [
            Citation(
                document_name=c.document_name or "Unknown",
                page=c.page,
                similarity_score=c.similarity_score,
                chunk_id=c.chunk_id,
                exact_snippet=c.text[:120] + "..."
            ) for c in retrieved_chunks
        ]

        return Answer(
            text=text,
            is_faithful=len(verified_quotes) > 0,
            citations=citations,
            confidence_score=confidence_score,
            confidence_level=confidence_level
        )

    @staticmethod
    def _extract_json_list(raw_string: str, field_name: str) -> List[str]:
        if not raw_string:
            return []
            
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_string, re.DOTALL)
        if json_match: 
            raw_string = json_match.group(1)
            
        # Fallback if LLM returned a direct array instead of object
        list_match = re.search(r'\[.*?\]', raw_string, re.DOTALL)
        dict_match = re.search(r'\{.*?\}', raw_string, re.DOTALL)

        try:
            if dict_match:
                data = json.loads(dict_match.group(0))
                return data.get(field_name, [])
            elif list_match and not dict_match:
                # LLM bypassed the object format and just returned an array
                return json.loads(list_match.group(0))
        except json.JSONDecodeError:
            pass
            
        return []