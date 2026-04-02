## How it works

Papers (PDF) are placed in `papers/{year}/`. The pipeline runs in three steps:

1. **Extract** — PyMuPDF scans each PDF block-by-block for an *Impact Statement* or *Broader Impact* header and grabs the text that follows. If nothing is found, a FAISS RAG search over impact-related chunks is used as fallback. If that also fails, the full paper text is sent.

2. **Analyse** — The extracted text is passed to `gpt-4o-mini` via a LangChain chain with `with_structured_output`, which enforces a strict Pydantic schema and answers four questions:
   - **Q1** — Is the impact statement a generic default phrase?
   - **Q2** — Does the paper describe positive societal/environmental impact?
   - **Q3** — Does the paper describe negative societal/environmental impact?
   - **Q4** — Do positive impacts outweigh negative ones?

3. **Save** — Results are written as `results/{year}/{paper}.json` and aggregated into `results/summary.json`, served by FastAPI at `/data` and visualised at `/`.

---

## INSTALLATION

### WINDOW
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### LINUX
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Sync VENV
```bash
uv sync
source .venv/bin/activate
```