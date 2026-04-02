import re
import json
import os
import argparse

import fitz
from dotenv import load_dotenv
from pathlib import Path
from typing import List

from pydantic import Field, BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

_openai_key = os.getenv("OPENAI_API_KEY")
if _openai_key:
    os.environ["OPENAI_API_KEY"] = _openai_key

# llm = ChatOllama(model="qwen3.5:4b", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) if _openai_key else None

# embeddings = OllamaEmbeddings(model="qwen3.5:4b")
embeddings = OpenAIEmbeddings() if _openai_key else None

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 1000,
    chunk_overlap = 200,
    separators    = ["\n\n", "\n", ". ", " "],
)

IMPACT_PATTERNS = [
    r"broader\s+impact",
    r"societal\s+impact",
    r"impact\s+statement",
    r"ethics\s+statement",
    r"ethical\s+(issues|considerations?|discussion)",
    r"risks?\s+and\s+limitations",
    r"limitations?\s+and\s+(broader|societal|ethical)",
    r"negative\s+societal",
    r"positive\s+societal",
    r"social\s+consequence",
]

IMPACT_HEADING = re.compile(
    r"broader\s+impacts?|"
    r"societal\s+impacts?|"
    r"impact\s+statement|"
    r"ethics\s+statement|"
    r"ethical\s+(issues|considerations?)|"
    r"risks?\s+and\s+limitations|"
    r"limitations?\s+(and\s+)?(broader|societal|ethical)|"
    r"broader\s+context",
    re.IGNORECASE,
)


class ImpactAnalysis(BaseModel):
    q1_default_phrase  : bool      = Field(description="True if the impact statement is generic/default with no specific meaning")
    q2_positive_impact : bool      = Field(description="True if the article describes positive societal or environmental impact")
    q3_negative_impact : bool      = Field(description="True if the article describes negative societal or environmental impact")
    q4_more_positive   : bool      = Field(description="True if positive impacts outweigh negative impacts")
    positive_examples  : List[str] = Field(description="Short list of positive impact examples found")
    negative_examples  : List[str] = Field(description="Short list of negative impact examples found")
    impact_text_found  : bool      = Field(description="True if an impact statement section was found in the PDF")


class RunRequest(BaseModel):
    papers : str = Field(default="papers",  description="Folder containing year subfolders with PDFs")
    output : str = Field(default="results", description="Folder to save JSON results")


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert in AI ethics and societal impact evaluation. "
        "You MUST reason carefully from the provided text to answer every question. "
        "Even if no explicit impact statement section exists, infer from the paper content. "
        "ALWAYS provide concrete examples in positive_examples and negative_examples — "
        "never leave them empty if the paper has any real-world application or risk. "
        "Do not fabricate, but do reason from what is written.",
    ),
    (
        "human",
        "Paper title: {title}\n\n"
        "Text (impact statement or full paper excerpts):\n{impact_text}\n\n"
        "Answer ALL four questions:\n\n"
        "Q1: Is the impact statement a generic default phrase with no specific meaning? "
        "(e.g. 'This paper presents work whose goal is to advance the field of machine learning. "
        "There are many potential societal consequences of our work, none of which we feel must be "
        "specifically highlighted here.') Answer YES or NO.\n\n"
        "Q2: Does this paper have any POSITIVE societal or environmental impact? "
        "(e.g. improves healthcare, reduces energy use, improves fairness, benefits society) "
        "Answer YES or NO, and list specific examples in positive_examples.\n\n"
        "Q3: Does this paper have any NEGATIVE societal or environmental impact? "
        "(e.g. misuse risk, privacy risk, energy cost, bias amplification, job displacement) "
        "Answer YES or NO, and list specific examples in negative_examples.\n\n"
        "Q4: Do the positive impacts outweigh the negative impacts overall? Answer YES or NO.\n\n"
        "Be specific — extract real phrases or applications from the text as examples.",
    ),
])

chain = (prompt | llm.with_structured_output(ImpactAnalysis)) if llm else None


def extract_impact_section(pdf_path: Path) -> str:
    doc          = fitz.open(str(pdf_path))
    text_results = ""
    grab_next    = 0

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            if grab_next:
                text_results += "\n" + text
                grab_next    -= 1
            if IMPACT_HEADING.search(text) and len(text) < 120:
                text_results += "\n" + text
                grab_next     = 5

    return text_results.strip()


def extract_full_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    return "\n".join(
        block[4].strip()
        for page in doc
        for block in page.get_text("blocks")
        if block[4].strip()
    )


def build_rag_context(pdf_path: Path, title: str) -> str:
    loader   = PyPDFLoader(str(pdf_path))
    pages    = loader.load()
    chunks   = splitter.split_documents(pages)
    relevant = [
        c for c in chunks
        if any(re.search(p, c.page_content.lower()) for p in IMPACT_PATTERNS)
    ]

    if not relevant:
        return ""

    store   = FAISS.from_documents(relevant, embeddings)
    results = store.similarity_search(
        f"societal impact ethical consequences broader impact statement of {title}",
        k = 6,
    )
    return "\n\n".join(r.page_content for r in results)


def analyse_paper(pdf_path: Path) -> dict:
    impact_section = extract_impact_section(pdf_path)
    rag_context    = build_rag_context(pdf_path, pdf_path.stem) if not impact_section else ""
    full_text      = extract_full_text(pdf_path) if not impact_section and not rag_context else ""
    impact_text    = impact_section or rag_context or full_text or "No impact statement found."
    found          = bool(impact_section or rag_context)

    if chain is None:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot analyse papers.")

    result = chain.invoke({
        "title"      : pdf_path.stem,
        "impact_text": impact_text,
    })

    return {
        "title"            : pdf_path.stem,
        "q1_default"       : result.q1_default_phrase,
        "q2_positive"      : result.q2_positive_impact,
        "q3_negative"      : result.q3_negative_impact,
        "q4_more_positive" : result.q4_more_positive,
        "positive_examples": result.positive_examples,
        "negative_examples": result.negative_examples,
        "impact_found"     : found,
    }


def analyse_folder(papers_dir: Path, output_dir: Path) -> List[dict]:
    year_dirs = sorted([d for d in papers_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    if not year_dirs:
        print(f"No year subfolders found in {papers_dir}. Expected: papers/2021/, papers/2022/ …")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records : List[dict] = []

    for year_dir in year_dirs:
        year       = year_dir.name
        pdf_files  = sorted(year_dir.glob("*.pdf"))
        year_out   = output_dir / year
        year_out.mkdir(parents=True, exist_ok=True)
        year_records : List[dict] = []

        print(f"\n[{year}] {len(pdf_files)} PDF(s)")

        for pdf_path in pdf_files:
            out = year_out / f"{pdf_path.stem}.json"

            if out.exists():
                record = json.loads(out.read_text(encoding="utf-8"))
                year_records.append(record)
                all_records.append(record)
                print(f"  Skipping   : {pdf_path.name} (already done)")
                continue

            print(f"  Processing : {pdf_path.name}")
            try:
                record         = analyse_paper(pdf_path)
                record["year"] = year
                year_records.append(record)
                all_records.append(record)

                out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  Saved      : {out.name}")
            except Exception as e:
                print(f"  [ERROR] {pdf_path.name} : {e}")

        if year_records:
            year_summary = year_out / "summary.json"
            year_summary.write_text(json.dumps(year_records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Year summary : {year_summary}")

    if all_records:
        global_summary = output_dir / "summary.json"
        global_summary.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Global summary : {global_summary}")

    print(f"\nDone. {len(all_records)} papers analysed → {output_dir}/")
    return all_records


app = FastAPI()

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/data")
def data():
    summary = Path("results/summary.json")
    if not summary.exists():
        raise HTTPException(status_code=404, detail="No results yet. Run /run first.")
    return json.loads(summary.read_text(encoding="utf-8"))


@app.post("/run")
def run(req: RunRequest):
    papers_dir = Path(req.papers)
    output_dir = Path(req.output)

    if not papers_dir.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {papers_dir}")

    records = analyse_folder(papers_dir, output_dir)

    if not records:
        raise HTTPException(status_code=404, detail="No PDFs found or processed.")

    return {
        "processed" : len(records),
        "output_dir": str(output_dir),
        "results"   : records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse societal impact of AI papers from a folder of PDFs."
    )
    parser.add_argument("papers", type=Path,                          help="Folder containing PDF files")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Folder to save results")

    args = parser.parse_args()

    analyse_folder(args.papers, args.output)
