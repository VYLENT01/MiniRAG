"""
Text Cleaner Implementation.

Performs basic normalization. 
Educational note: Aggressive cleaning (like removing all punctuation) 
can harm embedding quality. We stick to whitespace normalization here.
"""

import re
from minirag.cleaners.base import BaseCleaner


class TextCleaner(BaseCleaner):
    """Cleans raw text by normalizing whitespace and fixing common PDF artifacts."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Replace various whitespace characters (tabs, non-breaking spaces) with a standard space
        text = re.sub(r'[\t\x00-\x1f\x7f-\x9f\u00a0]+', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove spaces at the start and end of lines
        lines = [line.strip() for line in text.splitlines()]
        
        # Join lines back, collapsing more than 2 consecutive newlines into 2
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()