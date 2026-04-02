import re
import json
import argparse
import os

from dotenv import load_dotenv
from pathlib import Path
from typing import List

import openreview
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pydantic import Field, BaseModel
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# llm = ChatOllama(model="qwen3:latest", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# embeddings = OllamaEmbeddings(model="qwen3:latest")
embeddings = OpenAIEmbeddings()

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 1000,
    chunk_overlap = 200,
    separators    = ["\n\n", "\n", ". ", " "],
)

IMPACT_PATTERNS = [
    r"broader\s+impact",
    r"societal\s+impact",
    r"ethical\s+consideration",
    r"social\s+consequence",
    r"potential\s+impact",
    r"impact\s+statement",
    r"negative\s+societal",
    r"positive\s+societal",
]

DEFAULT_PHRASE = (
    "this paper presents work whose goal is to advance the field of machine learning"
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
    venue_id : str  = Field(description="OpenReview venue ID (e.g. NeurIPS.cc/2023/Conference)")
    limit    : int  = Field(default=20, description="Max number of PDFs to download")
    papers   : str  = Field(default="papers",  description="Folder to save downloaded PDFs")
    output   : str  = Field(default="results", description="Folder to save JSON results")


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert in AI ethics and societal impact evaluation. "
        "Answer strictly based on the provided impact statement text. "
        "Be concise and precise. Do not invent information not present in the text.",
    ),
    (
        "human",
        "Paper title: {title}\n\n"
        "Impact statement / relevant excerpts:\n{impact_text}\n\n"
        "Answer the four questions:\n"
        "Q1: Does the impact statement use only a generic default phrase with no specific meaning "
        "(e.g. 'This paper presents work whose goal is to advance the field of machine learning. "
        "There are many potential societal consequences of our work, none of which we feel must be "
        "specifically highlighted here.') or equivalent? YES/NO\n"
        "Q2: Does the article describe any POSITIVE societal or environmental impact? YES/NO\n"
        "Q3: Does the article describe any NEGATIVE societal or environmental impact? YES/NO\n"
        "Q4: Do the positive impacts outweigh the negative impacts? YES(P>N)/NO\n\n"
        "Also list concrete positive and negative examples if any.",
    ),
])

chain = prompt | llm.with_structured_output(ImpactAnalysis)


def extract_impact_section(pages: List[Document]) -> str:
    full_text    = "\n".join(p.page_content for p in pages)
    lines        = full_text.split("\n")
    impact_lines : List[str] = []
    in_section   = False

    for line in lines:
        lower = line.lower()
        if any(re.search(p, lower) for p in IMPACT_PATTERNS):
            in_section = True
        if in_section:
            impact_lines.append(line)
            if len(impact_lines) > 80:
                break

    return "\n".join(impact_lines) if impact_lines else ""


def build_rag_context(pages: List[Document], title: str) -> str:
    chunks   = splitter.split_documents(pages)
    relevant = [
        c for c in chunks
        if any(re.search(p, c.page_content.lower()) for p in IMPACT_PATTERNS)
    ]

    if not relevant:
        return ""

    store   = FAISS.from_documents(relevant, embeddings)
    results = store.similarity_search(
        f"societal impact ethical consequences of {title}",
        k = 4,
    )
    return "\n\n".join(r.page_content for r in results)


def analyse_paper(pdf_path: Path) -> dict:
    loader = PyPDFLoader(str(pdf_path))
    pages  = loader.load()

    impact_section = extract_impact_section(pages)
    rag_context    = build_rag_context(pages, pdf_path.stem)
    impact_text    = impact_section or rag_context or "No impact statement found."
    found          = bool(impact_section or rag_context)

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


def download_from_openreview(venue_id: str, output_dir: Path, limit: int = 20) -> None:
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
    output_dir.mkdir(parents=True, exist_ok=True)

    submissions = client.get_all_notes(
        invitation = f"{venue_id}/-/Submission",
        details    = "directReplies",
    )

    downloaded = 0
    for note in submissions:
        if downloaded >= limit:
            break

        pdf_url = note.content.get("pdf", {})
        if isinstance(pdf_url, dict):
            pdf_url = pdf_url.get("value", "")
        if not pdf_url:
            continue

        title    = re.sub(r'[^\w\s-]', '', note.content.get("title", {}).get("value", note.id))
        title    = title.strip().replace(" ", "_")[:80]
        out_path = output_dir / f"{title}.pdf"

        if out_path.exists():
            downloaded += 1
            continue

        try:
            pdf_bytes = client.get_attachment(note.id, "pdf")
            out_path.write_bytes(pdf_bytes)
            print(f"  Downloaded : {out_path.name}")
            downloaded += 1
        except Exception as e:
            print(f"  [SKIP] {note.id} : {e}")

    print(f"\n  Total downloaded : {downloaded} PDF(s) → {output_dir}/")


def run_batch(pdf_dir: Path, output_dir: Path) -> List[dict]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []

    for pdf_path in pdf_files:
        print(f"  Processing : {pdf_path.name}")
        try:
            record = analyse_paper(pdf_path)
            records.append(record)

            out = output_dir / f"{pdf_path.stem}.json"
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Saved      : {out.name}")
        except Exception as e:
            print(f"  [ERROR] {pdf_path.name} : {e}")

    if records:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Summary    : {summary_path}")

    return records


def visualize(records: List[dict], output_path: Path) -> None:
    titles = [r["title"][:40] for r in records]
    q1     = [int(r["q1_default"])       for r in records]
    q2     = [int(r["q2_positive"])      for r in records]
    q3     = [int(r["q3_negative"])      for r in records]
    q4     = [int(r["q4_more_positive"]) for r in records]

    total  = len(records)
    labels = ["Q1 Default phrase", "Q2 Positive impact", "Q3 Negative impact", "Q4 Positive > Negative"]
    counts = [sum(q1), sum(q2), sum(q3), sum(q4)]
    colors = ["#e07b54", "#5aab6e", "#d94f4f", "#5b8dd9"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Societal Impact Analysis — NeurIPS Papers", fontsize=14, fontweight="bold")

    ax1  = axes[0]
    bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax1.set_title(f"Aggregate Results (n={total})", fontsize=11)
    ax1.set_ylabel("Number of papers")
    ax1.set_ylim(0, total + 1)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.tick_params(axis="x", labelsize=8)
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(count),
            ha         = "center",
            va         = "bottom",
            fontsize   = 9,
            fontweight = "bold",
        )

    ax2   = axes[1]
    x     = range(len(titles))
    width = 0.2
    ax2.bar([i - 1.5 * width for i in x], q1, width, label="Q1 Default",  color=colors[0])
    ax2.bar([i - 0.5 * width for i in x], q2, width, label="Q2 Positive", color=colors[1])
    ax2.bar([i + 0.5 * width for i in x], q3, width, label="Q3 Negative", color=colors[2])
    ax2.bar([i + 1.5 * width for i in x], q4, width, label="Q4 P>N",      color=colors[3])
    ax2.set_title("Per-paper Breakdown", fontsize=11)
    ax2.set_ylabel("YES (1) / NO (0)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(titles, rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(0, 1.4)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved : {output_path}")


app = FastAPI()


@app.post("/run")
def run(req: RunRequest):
    papers_dir = Path(req.papers)
    output_dir = Path(req.output)

    try:
        download_from_openreview(req.venue_id, papers_dir, req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    records = run_batch(papers_dir, output_dir)

    if not records:
        raise HTTPException(status_code=404, detail="No PDFs found or processed.")

    return {
        "processed" : len(records),
        "output_dir": str(output_dir),
        "results"   : records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download NeurIPS papers from OpenReview and analyse their societal impact."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dl = subparsers.add_parser("download", help="Download PDFs from OpenReview.")
    dl.add_argument("venue_id", type=str,  help="OpenReview venue ID (e.g. NeurIPS.cc/2023/Conference).")
    dl.add_argument("--output", type=Path, default=Path("papers"),  help="Folder to save PDFs.")
    dl.add_argument("--limit",  type=int,  default=20,              help="Max number of PDFs to download.")

    an = subparsers.add_parser("analyse", help="Analyse PDFs already downloaded.")
    an.add_argument("pdf_dir",  type=Path,                          help="Folder containing PDF files.")
    an.add_argument("--output", type=Path, default=Path("results"), help="Folder to save JSON results.")

    full = subparsers.add_parser("run", help="Download then analyse in one step.")
    full.add_argument("venue_id", type=str,  help="OpenReview venue ID.")
    full.add_argument("--papers", type=Path, default=Path("papers"),  help="Folder to save PDFs.")
    full.add_argument("--output", type=Path, default=Path("results"), help="Folder to save results.")
    full.add_argument("--limit",  type=int,  default=20,              help="Max number of PDFs to download.")

    args = parser.parse_args()

    if args.command == "download":
        download_from_openreview(args.venue_id, args.output, args.limit)

    elif args.command == "analyse":
        run_batch(args.pdf_dir, args.output)

    elif args.command == "run":
        download_from_openreview(args.venue_id, args.papers, args.limit)
        run_batch(args.papers, args.output)
