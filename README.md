# Citation Finder Agent

A lightweight command-line tool that automatically detects factual claims in your text, retrieves relevant scholarly references via OpenAlex, and annotates your document with inline citations and a generated bibliography.

---

## Features

- **Claim Detection**  
  Uses a tuned LLM prompt to identify factual claims within each sentence and extract minimal “claim spans.”

- **Query Generation**  
  Converts each claim span into concise search queries tailored for scholarly discovery.

- **Reference Retrieval**  
  Leverages the [OpenAlex API](https://openalex.org/) to fetch candidate papers for each query.

- **Candidate Reranking**  
  Summarizes paper abstracts via LLM and ranks them by relevance to each claim.

- **Multi-Citation Support**  
  Attaches multiple high-scoring references to each claim, rather than a single “best” match.

- **Automatic Annotation**  
  Inserts inline citations (e.g. `(Smith et al., 2020; Doe et al., 2018)`) and compiles a “References” section at the end of your document.

- **Configurable Parameters**  
  Control maximum candidates, top-K citations, retry logic, LLM model choice, verbosity, and more via `.env`.

- **Easy to Extend**  
  Modular architecture—swap out LLM providers, retrieval backends, or tuning prompts with minimal code changes.

---

## Installation
Ensure that you have uv installed on your system!

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/tu-llm-agent.git
   cd tu-llm-agent
2. **Set up the virtual environment using uv**
   ```bash
   uv sync
4. **Copy and populate environment variables**
   ```bash
   cp .env.example .env

