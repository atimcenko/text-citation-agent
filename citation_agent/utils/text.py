import nltk
from typing import List

# Download tokenizer data if not present
nltk.download('punkt_tab', quiet=True)

def split_sentences(text: str) -> List[str]:
    """
    Split a paragraph into sentences using NLTK's Punkt tokenizer.
    """
    cleaned = text.strip().replace('\n', ' ')
    return nltk.sent_tokenize(cleaned)