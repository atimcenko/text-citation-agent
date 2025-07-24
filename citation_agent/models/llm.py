from __future__ import annotations
import os
import time
import json

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import ContextWindowExceededError

# Load .env
load_dotenv()

# Configuration
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", 2))
RETRY_DELAY   = float(os.getenv("RETRY_DELAY", 1.0))
VERBOSE       = os.getenv("VERBOSE", "0") == "1"
MAX_CAND      = int(os.getenv("MAX_CAND", "25"))
TOP_K         = int(os.getenv("TOP_K", "10"))
NUM_QUERIES   = int(os.getenv("NUM_QUERIES", "5"))

def set_verbose(v: bool):
    """Enable or disable verbose debug printing."""
    global VERBOSE
    VERBOSE = v

def llm_call(prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Send a prompt to LiteLLM and return the assistant's reply text.
    Retries on failure. Prints prompt & response if VERBOSE.
    """
    if VERBOSE and False:
       print("⏳ [LLM] Prompt:\n", prompt, "\n" + "-"*40)

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = completion(
                model=LITELLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp["choices"][0]["message"]["content"]
            if VERBOSE:
                print("✅ [LLM] Response:\n", text, "\n" + "="*40)
            return text
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES+1} attempts: {e}")

# --- detect_claims --------------------------------------------------------

_DETECT_FEWSHOT = """
Sentence: "Water boils at 100°C at sea level."
{"needs_cite": false, "claim_spans": []}

Sentence: "The hippocampus supports spatial memory in mammals."
{"needs_cite": true, "claim_spans": ["The hippocampus supports spatial memory in mammals"]}

Sentence: "Representational similarity analysis (RSA) is widely used to compare neural state patterns."
{"needs_cite": true, "claim_spans": ["Representational similarity analysis (RSA) is widely used to compare neural state patterns"]}

Sentence: "In V1 direction sensitive cells were found, serving as frequency selective filters, which allowed to construct convolutional neural network models of the brain."
{"needs_cite": true, "claim_spans": ["In V1 direction sensitive cells were found", "which allowed to construct convolutional neural network models of the brain"]}
"""

_DETECT_PROMPT = """
You are a scientific writing assistant.
Input: a single sentence.

Example sentences which require claims already with citations:

Neural state space diagrams can have as many axes as there are recorded neurons [no evidence needed here].
However, neural activity often only varies along a smaller number of directions in the state space [supporting evidence needed here], known as dimensions
Because activities of different neurons are correlated with one another [supporting evidence needed here]
every neuron does not make an independent contribution to the population [supporting evidence needed here]

Task:
1. Split the sentence into independent factual claims.
2. Decide for each claim if it needs a scholarly citation.
3. Remember, your main task is to help avoid making unsupported claims! Think carefully and decide whether a claim needs support for the reader to verify it.
4. Return JSON with keys exactly:
   needs_cite: true|false
   claim_spans: [ ... list of claim strings needing citation ... ]


Sentence:
\"\"\"{sentence}\"\"\"
"""

def detect_claims(sentence: str) -> dict:
    prompt = _DETECT_FEWSHOT + _DETECT_PROMPT.format(sentence=sentence)
    resp = llm_call(prompt, max_tokens=250)
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        # fallback heuristic
        keywords = ("suggests that", "we found", "demonstrated",
                    "previous studies", "has been shown",
                    "widely used", "first developed")
        needs = any(kw in sentence.lower() for kw in keywords)
        spans = [sentence] if needs else []
        return {"needs_cite": needs, "claim_spans": spans}

# --- gen_queries ----------------------------------------------------------

_GENQ_PROMPT = """
You are a scholarly-query generator.
Given a claim span and its full sentence, produce {n} concise lowercase search queries that would help find papers supporting the claim.
Keep in mind the domain specificity. You can try to infer the domain from the claim. Use domain-specific keywords in such case.
In case the domain is not clear, create generic search requests relevant for clarifying the clame faithfullness.
In case you think the domain is well-defined based on the sentence and the claim span, try using your prior knowledge about the authors and early works influential in the domain.
For example, if the claim references transformer architecture the search query could be "transformers vaswani", "attention is all you need"
In case the voltage-gated sodium channels are mentioned one could search for "hodgkin huxley squid axon conductances", "voltage gated sodium channels review" etc.

Return JSON with exactly, no brackets, not "json" markers. Pure JSON exactly like this:
{{
  "queries": [ "query1", "query2", "query3", "query4", "query5"]
}}

Eeach query must be unique and tackle the statement from different perspectives.

Claim span:
\"\"\"{claim}\"\"\"

Full sentence:
\"\"\"{sentence}\"\"\"
""".replace("{n}", str(NUM_QUERIES))

def gen_queries(claim: str, sentence: str) -> list[str]:
    prompt = _GENQ_PROMPT.format(claim=claim, sentence=sentence)
    resp = llm_call(prompt, max_tokens=200, temperature=1)
    try:
        raw = json.loads(resp)["queries"]
        # preserve order, drop exact duplicates
        queries: list[str] = []
        for q in raw:
            q = q.strip()
            if q and q not in queries:
                queries.append(q)
        return queries
    except (json.JSONDecodeError, KeyError):
        raise ValueError(f"Invalid JSON from gen_queries:\n{resp}")

# --- summarize_abstracts -------------------------------------------------

_SUMMARIZE_FEWSHOT = """
Abstract: [skipped to main factual claim] We trained a deep neural network on ImageNet, achieving state-of-the-art results.
Summary: A deep neural network trained on ImageNet achieved state-of-the-art performance.

Abstract key sentence: [abstract beginning]...This is the key sentence or sentences you need to identify-->.In this study, we propose a novel clustering algorithm that improves speed by 30%.
Summary: A novel clustering algorithm is proposed that improves speed by 30%.
"""

_SUMMARIZE_PROMPT = """
You are an expert at reading scientific abstracts.
Given the abstract below, identify key factual claims that the authors make. What is their main contribution? 
Write ONE clear sentence (≤30 words) capturing the main finding or method.
Do NOT start with 'This paper', begin directly with the result.

Abstract:
\"\"\"{abstract}\"\"\"
"""

def summarize_abstracts(candidates: list[dict]) -> list[dict]:
    summarized: list[dict] = []
    for c in candidates[:MAX_CAND]:
        abs_text = c.get("abstract", "") or ""
        if abs_text:
            prompt = _SUMMARIZE_FEWSHOT + _SUMMARIZE_PROMPT.format(abstract=abs_text)
            summary = llm_call(prompt, max_tokens=60, temperature=0.0)
            summary = " ".join(summary.strip().split())
        else:
            summary = ""
        entry = c.copy()
        entry["summary"] = summary
        summarized.append(entry)
    return summarized

# --- rerank ---------------------------------------------------------------

_RERANK_PROMPT_HEAD = """
You are an expert at matching scholarly works to claims.
Below is the claim and the sentence it comes from.
Next is a list of candidate papers (id, title, summary of findings).
For each candidate, assign a relevance score 0–5 (5=strong support).
Return exactly a JSON array:
[
  {{ "id": "<paper-id>", "score": <0–5> }},
  ...
]
"""

def rerank(claim: str, sentence: str, candidates: list[dict], top_k: int = TOP_K, min_score: int = 4) -> list[dict]:
    # Summarize up to MAX_CAND
    cands = summarize_abstracts(candidates)

    # Build prompt snippet
    snippet = [{"id": c.get("doi","") or c.get("id",""),
                "title": c["title"],
                "summary": c["summary"]} for c in cands]

    prompt = _RERANK_PROMPT_HEAD
    prompt += f"\nClaim:\n\"\"\"{claim}\"\"\"\n"
    prompt += f"\nSentence:\n\"\"\"{sentence}\"\"\"\n"
    prompt += "\nCandidates:\n" + json.dumps(snippet, indent=2)

    try:
        resp = llm_call(prompt, max_tokens=600)
        scores = json.loads(resp)
        score_map = {it["id"]: it["score"] for it in scores}
        # collect all above threshold
        selected = [c.copy() for c in candidates if score_map.get(c.get("doi","") or c.get("id",""),0) >= min_score]
        # if none, fall back to top_k by score
        if not selected:
            ranked = sorted(
                [{**c, "score": score_map.get(c.get("doi","") or c.get("id",""),0)} for c in candidates],
                key=lambda x: x["score"], reverse=True
            )
            selected = ranked[:top_k]
        else:
            # attach scores and sort
            for c in selected:
                c["score"] = score_map.get(c.get("doi","") or c.get("id",""),0)
            selected.sort(key=lambda x: x["score"], reverse=True)
        return selected
    except (ContextWindowExceededError, json.JSONDecodeError):
        # fallback keyword-match
        keywords = set(claim.lower().split())
        scored = []
        for c in candidates:
            text = (c["title"] + " " + c.get("summary","")).lower()
            score = sum(1 for kw in keywords if kw in text)
            entry = c.copy(); entry["score"] = score
            scored.append(entry)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
