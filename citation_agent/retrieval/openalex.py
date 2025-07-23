"""OpenAlex retrieval helper functions.

We prefer OpenAlex because it is public and gives a citation graph, but you
can swap in Semantic Scholar easily by satisfying the same function
signatures.  All functions are pure (no global state) and raise exceptions
up to the caller so the agent loop can decide whether to retry or fall
back.
"""

from __future__ import annotations
import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

OPENALEX_BASE_URL = os.getenv(
    "OPENALEX_BASE_URL",
    "https://api.openalex.org/works"
)
# Your email for polite OpenAlex usage; override via .env or env var
OPENALEX_EMAIL = os.getenv(
    "OPENALEX_EMAIL",
    "your_email@example.com"
)

print(OPENALEX_EMAIL)

try:
    DEFAULT_TOP_N = int(os.getenv("OPENALEX_DEFAULT_TOP_N", "5"))
except ValueError:
    DEFAULT_TOP_N = 5

def search_openalex(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Search OpenAlex works endpoint.
    Returns raw work dicts (up to top_n).
    """
    params = {
        'search': query,
        'per_page': top_n,
        'mailto': OPENALEX_EMAIL
    }
    resp = requests.get(OPENALEX_BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get('results', [])

def extract_abstract_inverted(inv_idx: Dict[str, List[int]]) -> str:
    """
    Reconstruct abstract text from OpenAlex's inverted index:
    { token: [pos, pos, ...], ... }
    """
    if not inv_idx:
        return ''
    # Determine full length
    max_pos = max((pos for poses in inv_idx.values() for pos in poses), default=-1)
    tokens: List[str] = [''] * (max_pos + 1)
    for token, positions in inv_idx.items():
        for pos in positions:
            tokens[pos] = token
    return ' '.join(tokens)

def extract_abstract(work: Dict[str, Any]) -> str:
    """
    Wrapper: pull inverted index from a work and reconstruct.
    """
    inv = work.get('abstract_inverted_index', {})
    return extract_abstract_inverted(inv)

def get_top_references(query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for `query`, reconstruct abstracts, and return
    a list of dicts:
      { doi, title, authors, year, abstract }
    """
    works = search_openalex(query, top_n)
    refs: List[Dict[str, Any]] = []
    for w in works:
        doi = w.get('doi', '')
        title = w.get('title', '')
        year = w.get('publication_year')
        # authorship list -> list of author names
        authors = [
            a.get('author', {}).get('display_name', '')
            for a in w.get('authorships', [])
        ]
        abstract = extract_abstract(w)
        refs.append({
            'doi': doi,
            'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract
        })
    return refs
