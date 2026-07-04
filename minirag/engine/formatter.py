"""
Deterministic Answer Formatter with Comprehensive Debug Tracing.
"""

import json
import re
from typing import List

from minirag.models.answer import Answer, Citation, DebugTrace, RejectionReason
from minirag.models.search import SearchResult


class AnswerFormatter:
    """Assembles answers strictly from verified verbatim quotes and logs rejections."""

    @staticmethod
    def normalize_persian(text: str) -> str:
        """Strips diacritics, normalizes ZWNJ, characters, and removes punctuation."""
        diacritics_re = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
        text = diacritics_re.sub('', text)
        text = text.replace('\u200c', ' ') # ZWNJ to space
        text = text.replace('ی', 'ي').replace('ک', 'ك').replace('ة', 'ه')
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip().lower()

    @classmethod
    def format(cls, llm_raw_output: str, retrieved_chunks: List[SearchResult], threshold: float) -> Answer:
        trace = DebugTrace(
            llm_raw_json=llm_raw_output,
            retrieval_scores=[c.similarity_score for c in retrieved_chunks]
        )

        quotes = cls._extract_json_list(llm_raw_output, "quotes")
        trace.extracted_quotes = quotes
        
        chunk_texts_combined = " ".join([c.text for c in retrieved_chunks])
        clean_corpus = cls.normalize_persian(chunk_texts_combined)

        verified_quotes = []
        for quote in quotes:
            clean_quote = cls.normalize_persian(quote)
            
            if not clean_quote:
                trace.rejected_quotes.append(RejectionReason(quote, "Empty after normalization"))
            elif len(clean_quote) < 10:
                trace.rejected_quotes.append(RejectionReason(quote, f"Too short (len: {len(clean_quote)})"))
            elif clean_quote not in clean_corpus:
                trace.rejected_quotes.append(RejectionReason(quote, "Substring NOT found in corpus (Hallucination/Typo)"))
            else:
                verified_quotes.append(quote)

        # Calculate Confidence (Margin-based)
        if not retrieved_chunks:
            return Answer(text="پاسخ این سؤال در اسناد موجود یافت نشد.", is_faithful=False, confidence_score=0.0, confidence_level="NONE", citations=[], trace=trace)

        scores = sorted([c.similarity_score for c in retrieved_chunks], reverse=True)
        margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0
        confidence_score = scores[0] - (margin * 0.2)
        confidence_level = "HIGH" if confidence_score >= 0.6 else ("MEDIUM" if confidence_score >= 0.4 else "LOW")

        if not verified_quotes:
            text = "سیستم نتوانست پاسخ دقیقی از متن استخراج کند."
            confidence_level = "REJECTED"
            confidence_score = 0.0
        else:
            text = "\n".join(f"• {q}" for q in verified_quotes)

        citations = [
            Citation(document_name=c.document_name or "Unknown", page=c.page, similarity_score=c.similarity_score, chunk_id=c.chunk_id, exact_snippet=c.text[:120]+"...") 
            for c in retrieved_chunks
        ]

        return Answer(text=text, is_faithful=len(verified_quotes) > 0, citations=citations, confidence_score=confidence_score, confidence_level=confidence_level, trace=trace)

    @staticmethod
    def _extract_json_list(raw_string: str, field_name: str) -> List[str]:
        if not raw_string: return []
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_string, re.DOTALL)
        if json_match: raw_string = json_match.group(1)
        list_match = re.search(r'\[.*?\]', raw_string, re.DOTALL)
        dict_match = re.search(r'\{.*?\}', raw_string, re.DOTALL)

        try:
            if dict_match:
                data = json.loads(dict_match.group(0))
                return data.get(field_name, [])
            elif list_match and not dict_match:
                return json.loads(list_match.group(0))
        except json.JSONDecodeError:
            pass
        return []