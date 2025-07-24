#!/usr/bin/env python3
"""
Citation Finder Agent

Usage:
  uv run citation_agent/agent.py data/your_input.txt [--verbose]
  python citation_agent/agent.py data/your_input.txt [-v]

This will read <your_input>.txt, annotate each claim‐span with inline citations,
and write the result to data/your_input_output.txt.

Options:
  -v, --verbose    Print debug logging for LLM calls and retrieval steps.

Configuration is via environment variables documented in .env.example.
"""

import argparse
from pathlib import Path
from utils.text import split_sentences
from retrieval.openalex import get_top_references
from models.llm import (
    set_verbose,
    detect_claims,
    gen_queries,
    rerank
)

def process_paragraph(text: str, verbose: bool = False):
    sentences = split_sentences(text)
    output = []

    for s in sentences:
        if verbose:
            print(f"\n▶ [Sentence] {s}")

        tag = detect_claims(s)
        if verbose:
            print("   [Tag] ", tag)

        if not tag["needs_cite"]:
            if verbose:
                print("   → No citation needed")
            output.append({"sentence": s, "claims": []})
            continue

        spans = list(dict.fromkeys(tag["claim_spans"]))

        span_results = []
        for span in spans:
            if verbose:
                print(f"  [Claim Span] {span}")

            # 1) generate queries
            queries = gen_queries(span, s)
            if verbose:
                print("    [Queries] ", queries)

            # 2) retrieve candidates
            all_cands = []
            for q in queries:
                if verbose:
                    print(f"    [Search] '{q}'")
                try:
                    refs = get_top_references(q)
                    if not isinstance(refs, list):
                        raise RuntimeError(f"Expected list, got {type(refs)}")
                    if verbose:
                        print(f"      [Results] {len(refs)}")
                except Exception as e:
                    print(f"      ⚠️  OpenAlex search failed for '{q}': {e}")
                    refs = []
                all_cands.extend(refs)


            # 3) dedupe by DOI
            seen = set()
            unique = []
            for c in all_cands:
                doi = c.get("doi", "")
                if doi and doi not in seen:
                    seen.add(doi)
                    unique.append(c)
            if verbose:
                print(f"    [Dedupe] {len(unique)} unique candidates")

            # 4) rerank this span’s candidates
            top_cits = rerank(span, s, unique, top_k=5)
            if verbose:
                print("    [Top citations]:")
                for r in top_cits:
                    print(f"       • {r.get('doi','')} (score={r.get('score')})")

            span_results.append({
                "span": span,
                "citations": top_cits
            })

        output.append({
            "sentence": s,
            "claims": span_results
        })

    return output


def write_with_references(input_path: str, mapping: list[dict], output_path: str):
    """
    Writes a new text file where:
    - Each claim‐span in each sentence gets its inline citations inserted:
        (Author1 et al., Year; Author2 et al., Year; …)
    - At the end, a References section listing all unique DOI entries.
    """
    original = Path(input_path).read_text(encoding="utf-8")
    sentences = split_sentences(original)

    seen = {}      # citation_key -> doi
    annotated = []

    def annotate_sentence(s: str, claims: list[dict]) -> str:
        inserts = []
        for claim in claims:
            span = claim["span"]
            start = s.find(span)
            if start == -1:
                continue
            end = start + len(span)

            keys = []
            for c in claim["citations"]:
                authors = c.get("authors", [])
                author_last = authors[0].split()[-1] if authors else "Unknown"
                year = c.get("year") or "n.d."
                key = f"{author_last} et al., {year}"
                seen[key] = c.get("doi", "")
                keys.append(key)

            if not keys:
                continue

            insert_text = " (" + "; ".join(keys) + ")"
            inserts.append((start, end, insert_text))

        # apply in reverse order
        new_s = s
        for start, end, text in sorted(inserts, key=lambda x: x[0], reverse=True):
            new_s = new_s[:end] + text + new_s[end:]
        return new_s

    for item in mapping:
        s = item["sentence"].strip()
        if not item["claims"]:
            annotated.append(s)
        else:
            annotated.append(annotate_sentence(s, item["claims"]))

    body = " ".join(annotated)

    # Build References section
    ref_lines = ["\n\nReferences:"]
    for key, doi in seen.items():
        ref_lines.append(f"- {key}: DOI {doi}")
    refs = "\n".join(ref_lines)

    Path(output_path).write_text(body + refs, encoding="utf-8")
    print(f"Wrote annotated file to {output_path} with {len(seen)} references.")


def main():
    ap = argparse.ArgumentParser(description="Citation Finder Agent")
    ap.add_argument("input_file", help="Text file with paragraph(s)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print debug info")
    args = ap.parse_args()

    set_verbose(args.verbose)

    text = Path(args.input_file).read_text(encoding="utf-8")
    mapping = process_paragraph(text, verbose=args.verbose)

    out_path = Path(args.input_file).with_name(
        Path(args.input_file).stem + "_with_references.txt"
    )
    write_with_references(args.input_file, mapping, str(out_path))


if __name__ == "__main__":
    main()
