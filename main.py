from __future__ import annotations

import ast
import contextvars
import json
import math
import operator
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
LOG_PATH = BASE_DIR / "tool_log.txt"

load_dotenv(BASE_DIR / ".env")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_host: str = os.getenv("PINECONE_HOST", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "lab06-agentic-rag")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "ai_agents_pdf")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    domain_pdf_path: Path = _resolve_path(os.getenv("DOMAIN_PDF_PATH", "data/ai_agents.pdf"))
    top_k: int = int(os.getenv("PINECONE_TOP_K", "5"))
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "8"))


settings = Settings()

_embedding_model: SentenceTransformer | None = None
_pinecone_index: Any | None = None
_groq_client: Groq | None = None
_tool_trace: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "tool_trace", default=None
)
_citations: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "citations", default=None
)


app = FastAPI(
    title="Agentic RAG Lab 06",
    description="A production-style FastAPI app with Groq tool calling, Pinecone RAG, Tavily search, Wikipedia summaries, and tool-call logging.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    tools_used: list[str]
    tool_trace: list[dict[str, Any]]
    citations: list[dict[str, Any]]


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    pages_indexed: int
    namespace: str
    source_file: str


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model_name)
    return _embedding_model


def get_pinecone_index() -> Any:
    global _pinecone_index
    if _pinecone_index is None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env and restart the server.")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            _pinecone_index = pc.Index(host=settings.pinecone_host)
        else:
            _pinecone_index = pc.Index(settings.pinecone_index_name)
    return _pinecone_index


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to .env and restart the server.")
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _append_tool_trace(tool: str, tool_input: Any, output: str) -> None:
    trace = _tool_trace.get()
    if trace is None:
        return
    trace.append(
        {
            "tool": tool,
            "input": tool_input,
            "output_preview": output[:700] + ("..." if len(output) > 700 else ""),
        }
    )


def _append_citation(page: int, text: str, score: float | None = None) -> None:
    citations = _citations.get()
    if citations is None:
        return

    snippet = " ".join(text.split())[:360]
    citation = {
        "source": settings.domain_pdf_path.name,
        "page": page,
        "snippet": snippet,
        "score": round(score, 4) if isinstance(score, (int, float)) else None,
    }

    dedupe_key = (citation["source"], citation["page"], citation["snippet"][:80])
    existing = {
        (item.get("source"), item.get("page"), str(item.get("snippet", ""))[:80])
        for item in citations
    }
    if dedupe_key not in existing:
        citations.append(citation)


def _safe_log_value(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "/").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def log_tool_call(tool_name: str, input_value: Any, success: bool) -> None:
    status = "success" if success else "failed"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {tool_name} | query='{_safe_log_value(input_value)}' | {status}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = get_embedding_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in embeddings]


def chunk_page_text(text: str, page_number: int, max_words: int = 420, overlap: int = 50) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []
    chunks: list[dict[str, Any]] = []
    step = max(max_words - overlap, 1)
    for chunk_index, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + max_words]
        if not chunk_words:
            continue
        chunks.append(
            {
                "page": page_number,
                "chunk_index": chunk_index,
                "text": " ".join(chunk_words),
            }
        )
        if start + max_words >= len(words):
            break
    return chunks


def _extract_pages_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": page_index, "text": text})
    return pages


def _extract_pages_with_pymupdf(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError:
        return []

    pages = []
    document = fitz.open(str(pdf_path))
    try:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append({"page": page_index, "text": text})
    finally:
        document.close()
    return pages


def _parse_sidecar_text(sidecar_path: Path) -> list[dict[str, Any]]:
    raw_text = sidecar_path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"^===\s*Page\s+(\d+)\s*===\s*$",
            raw_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if not matches:
        return [{"page": 1, "text": raw_text}] if raw_text.strip() else []

    pages = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        text = raw_text[start:end].strip()
        if text:
            pages.append({"page": int(match.group(1)), "text": text})
    return pages


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages = _extract_pages_with_pypdf(pdf_path)
    if pages:
        return pages

    pages = _extract_pages_with_pymupdf(pdf_path)
    if pages:
        return pages

    sidecar_candidates = [
        pdf_path.with_suffix(".txt"),
        pdf_path.with_name(f"{pdf_path.stem}_ocr.txt"),
        pdf_path.with_name(f"{pdf_path.stem}_knowledge_base.txt"),
    ]
    for sidecar_path in sidecar_candidates:
        if sidecar_path.exists():
            return _parse_sidecar_text(sidecar_path)

    return []


def search_knowledge_base(query: str) -> str:
    """
    Search the internal AI agents PDF knowledge base stored in Pinecone.
    Use this when the user asks about AI agents, agentic RAG, tool calling, or content from the uploaded domain PDF.
    Do not use for arithmetic, current news, live web facts, or simple encyclopedia summaries.
    Returns relevant PDF chunks with page citations.
    """
    try:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        index = get_pinecone_index()
        vector = embed_texts([query])[0]
        response = index.query(
            vector=vector,
            top_k=settings.top_k,
            include_metadata=True,
            namespace=settings.pinecone_namespace,
        )
        payload = _to_plain_dict(response)
        matches = payload.get("matches", [])

        if not matches:
            result = "No matching chunks were found in the AI agents knowledge base."
            log_tool_call("search_knowledge_base", query, True)
            return result

        lines = ["Knowledge base results from the AI agents PDF:"]
        for position, match in enumerate(matches, start=1):
            metadata = match.get("metadata", {}) or {}
            page = int(metadata.get("page", 0) or 0)
            text = str(metadata.get("text", "")).strip()
            score = match.get("score")
            if page and text:
                _append_citation(page=page, text=text, score=score)
            score_text = f", score {float(score):.3f}" if isinstance(score, (int, float)) else ""
            lines.append(f"\n{position}. Page {page}{score_text}\n{text[:1200]}")

        result = "\n".join(lines)
        log_tool_call("search_knowledge_base", query, True)
        return result
    except Exception as exc:
        log_tool_call("search_knowledge_base", query, False)
        return f"Knowledge base search failed: {exc}"


def search_web(query: str) -> str:
    """
    Search the live web using Tavily.
    Use this for recent facts, news, current releases, changing information, or topics not covered by the uploaded PDF.
    Do not use this for arithmetic or when the user explicitly asks for a Wikipedia article summary.
    Returns a concise answer plus ranked web result snippets and source URLs.
    """
    try:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")

        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "include_answer": "basic",
                "include_raw_content": False,
                "max_results": 5,
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()

        lines = ["Live web search results:"]
        if payload.get("answer"):
            lines.append(f"\nAnswer summary:\n{payload['answer']}")
        for index, result in enumerate(payload.get("results", []), start=1):
            title = result.get("title", "Untitled result")
            url = result.get("url", "")
            content = result.get("content", "")
            lines.append(f"\n{index}. {title}\nURL: {url}\nSnippet: {content}")

        result_text = "\n".join(lines)
        log_tool_call("search_web", query, True)
        return result_text
    except Exception as exc:
        log_tool_call("search_web", query, False)
        return f"Web search failed: {exc}"


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_NAMES: dict[str, Any] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


class SafeMathEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if op_type is ast.Pow and abs(right) > 100:
            raise ValueError("Exponent is too large for the calculator tool.")
        return _BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unary operator {op_type.__name__} is not allowed.")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
            raise ValueError("Only approved math functions are allowed.")
        func = _ALLOWED_NAMES[node.func.id]
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in _ALLOWED_NAMES and isinstance(_ALLOWED_NAMES[node.id], (int, float)):
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name {node.id!r} is not allowed.")

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Expression element {type(node).__name__} is not allowed.")


def safe_calculate(expression: str) -> float:
    normalized = expression.replace("^", "**")
    tree = ast.parse(normalized, mode="eval")
    return SafeMathEvaluator().visit(tree)


def calculate(expression: str) -> str:
    """
    Evaluate a safe arithmetic expression.
    Use this for math, numeric conversions, exponents, percentages, and quick calculations.
    The expression may use +, -, *, /, //, %, **, parentheses, and approved math functions like sqrt().
    Do not use this for knowledge-base lookup, Wikipedia summaries, or current web facts.
    """
    try:
        expression = expression.strip()
        if not expression:
            raise ValueError("expression cannot be empty")
        result = safe_calculate(expression)
        log_tool_call("calculate", expression, True)
        return f"Calculation result: {expression} = {result}"
    except Exception as exc:
        log_tool_call("calculate", expression, False)
        return f"Calculation failed: {exc}"


WIKIPEDIA_HEADERS = {"User-Agent": "AgenticRAGLab06/1.0 (student project)"}


def _clean_wikipedia_topic(topic: str) -> str:
    cleaned = re.sub(r"\s+", " ", topic).strip(" .?\"'")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+(?:in|for)\s+(?:nlp|natural language processing)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" .?\"'")


def _wikipedia_topic_candidates(topic: str) -> list[str]:
    cleaned = _clean_wikipedia_topic(topic)
    candidates: list[str] = []
    lower = topic.lower()

    if "transformer" in lower and (
        "nlp" in lower or "natural language processing" in lower or "architecture" in lower
    ):
        candidates.append("Transformer (deep learning)")

    candidates.extend([topic.strip(), cleaned])
    if cleaned.lower().endswith(" architecture"):
        candidates.append(cleaned[: -len(" architecture")].strip())

    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate.lower() not in {item.lower() for item in unique}:
            unique.append(candidate)
    return unique


def _fetch_wikipedia_summary(title: str) -> dict[str, Any] | None:
    slug = quote(title.replace(" ", "_"), safe="()")
    response = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
        headers=WIKIPEDIA_HEADERS,
        timeout=15,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    if payload.get("type") == "disambiguation":
        return None
    return payload


def _search_wikipedia_title(topic: str) -> str | None:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        headers=WIKIPEDIA_HEADERS,
        params={
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "format": "json",
            "srlimit": 5,
        },
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("query", {}).get("search", [])
    if not results:
        return None

    topic_lower = topic.lower()

    def score(result: dict[str, Any]) -> tuple[int, int]:
        title = str(result.get("title", ""))
        title_lower = title.lower()
        relevance = 0
        if "transformer" in topic_lower and title_lower == "transformer (deep learning)":
            relevance += 100
        if "transformer" in topic_lower and "transformer" in title_lower:
            relevance += 20
        if "architecture" in topic_lower and "deep learning" in title_lower:
            relevance += 10
        return relevance, int(result.get("wordcount", 0) or 0)

    best = max(results, key=score)
    return str(best.get("title", "")) or None


def get_wikipedia_summary(topic: str) -> str:
    """
    Fetch the opening summary of a Wikipedia article, using Wikipedia search if the exact title is not found.
    Use this when the user explicitly asks for a Wikipedia summary or an encyclopedia-style overview of a stable topic.
    Do not use this for current events, recent releases, arithmetic, or questions that should be answered from the uploaded PDF.
    Returns the article title, opening extract, and canonical Wikipedia URL.
    """
    try:
        topic = topic.strip()
        if not topic:
            raise ValueError("topic cannot be empty")

        payload = None
        resolved_from = None
        for candidate in _wikipedia_topic_candidates(topic):
            payload = _fetch_wikipedia_summary(candidate)
            if payload:
                resolved_from = candidate
                break

        if payload is None:
            search_title = _search_wikipedia_title(topic)
            if search_title:
                payload = _fetch_wikipedia_summary(search_title)
                resolved_from = search_title

        if payload is None:
            raise RuntimeError(f"No Wikipedia article summary found for topic: {topic}")

        title = payload.get("title", topic)
        extract = payload.get("extract") or "No opening summary was returned for this article."
        url = payload.get("content_urls", {}).get("desktop", {}).get("page", "")
        resolution_note = f"Resolved topic: {resolved_from}\n" if resolved_from and resolved_from != topic else ""
        result = f"Wikipedia summary for {title}:\n{resolution_note}{extract}\nURL: {url}"
        log_tool_call("get_wikipedia_summary", topic, True)
        return result
    except Exception as exc:
        log_tool_call("get_wikipedia_summary", topic, False)
        return f"Wikipedia summary failed: {exc}"


TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "search_knowledge_base": search_knowledge_base,
    "search_web": search_web,
    "calculate": calculate,
    "get_wikipedia_summary": get_wikipedia_summary,
}


TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "search_knowledge_base": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The semantic search query for the AI agents PDF knowledge base.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "search_web": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The live web search query to send to Tavily.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "calculate": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A safe arithmetic expression, for example '2 ** 16' or '5 * 365'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    "get_wikipedia_summary": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The Wikipedia topic or natural-language article phrase, for example 'Transformer architecture in NLP' or 'Artificial intelligence'.",
            }
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}


def build_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (fn.__doc__ or "").strip(),
                "parameters": TOOL_PARAMETERS[name],
            },
        }
        for name, fn in TOOL_FUNCTIONS.items()
    ]


SYSTEM_PROMPT = f"""
You are an agentic RAG assistant for Lab 06, built for M Abdullah Fawad.
Choose tools deliberately:
- search_knowledge_base: use for the uploaded AI agents PDF and domain-specific questions.
- search_web: use for current or changing information, latest releases, news, or unknown facts outside the PDF.
- calculate: use for arithmetic, powers, percentages, date math, and numeric expressions.
- get_wikipedia_summary: use when the user explicitly requests a Wikipedia summary or encyclopedic overview.
You may call multiple tools if the question has multiple parts.
When knowledge-base snippets include page numbers, cite the pages in the answer.
If no tool is needed, answer directly.
Keep answers clear, concise, and useful for a lab demonstration.
""".strip()


def _tool_call_to_payload(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments or "{}",
        },
    }


def _parse_tool_args(tool_name: str, arguments: str) -> dict[str, Any]:
    if tool_name not in TOOL_PARAMETERS:
        return {}
    try:
        parsed = json.loads(arguments or "{}")
        if isinstance(parsed, dict):
            return _normalize_tool_args(tool_name, parsed)
    except json.JSONDecodeError:
        pass

    fallback_name = next(iter(TOOL_PARAMETERS[tool_name]["properties"].keys()))
    return {fallback_name: arguments}


def _normalize_tool_args(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    parameters = TOOL_PARAMETERS[tool_name]
    required = parameters.get("required", [])
    normalized = {key: value for key, value in arguments.items() if key in parameters["properties"]}

    if all(key in normalized for key in required):
        return normalized

    first_value = next((value for value in arguments.values() if value is not None), "")
    for required_name in required:
        normalized.setdefault(required_name, first_value)
    return normalized


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        result = f"Unknown tool: {tool_name}"
        _append_tool_trace(tool_name, arguments, result)
        return result
    try:
        result = TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as exc:
        result = f"{tool_name} failed before execution: {exc}"
    _append_tool_trace(tool_name, arguments, result)
    return result


def _extract_wikipedia_topic(question: str) -> str:
    patterns = [
        r"wikipedia\s+summary\s+(?:of|about|for)\s+(.+)",
        r"summary\s+(?:of|about|for)\s+(.+)",
        r"wikipedia\s+(?:article\s+)?(?:on|about|for)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?\"'")
    return question.strip(" .?\"'")


def _extract_math_expression(question: str) -> str:
    lower = question.lower()

    power_match = re.search(
        r"(\d+(?:\.\d+)?)\s+(?:raised\s+to\s+the\s+power\s+of|to\s+the\s+power\s+of|power\s+of)\s+(\d+(?:\.\d+)?)",
        lower,
    )
    if power_match:
        return f"{power_match.group(1)} ** {power_match.group(2)}"

    days_match = re.search(r"(\d+(?:\.\d+)?)\s+years?", lower)
    if "day" in lower and days_match:
        return f"{days_match.group(1)} * 365"

    arithmetic = re.findall(r"[\d\s+\-*/().%^]+", question)
    expression = max((item.strip() for item in arithmetic), key=len, default="")
    return expression.replace("^", "**") if expression else question


def _looks_like_math(question: str) -> bool:
    lower = question.lower()
    math_words = ["calculate", "raised", "power", "percent", "percentage", "days", "years", "sqrt"]
    return any(word in lower for word in math_words) or bool(re.search(r"\d+\s*[\+\-\*/%^]\s*\d+", question))


def _looks_like_current_web(question: str) -> bool:
    lower = question.lower()
    web_words = [
        "latest",
        "recent",
        "current",
        "today",
        "news",
        "released",
        "2025",
        "2026",
        "most recent",
    ]
    return any(word in lower for word in web_words)


def _looks_like_domain_kb(question: str) -> bool:
    lower = question.lower()
    kb_words = [
        "pdf",
        "uploaded",
        "knowledge base",
        "ai agent",
        "ai agents",
        "agentic",
        "rag",
        "react",
        "tool design",
        "tool calling",
        "pinecone",
        "vector",
    ]
    return any(word in lower for word in kb_words)


def synthesize_answer_from_observations(question: str, observations: list[dict[str, str]]) -> str:
    observation_text = "\n\n".join(
        f"Tool: {item['tool']}\nObservation:\n{item['result']}" for item in observations
    )
    try:
        completion = get_groq_client().chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user clearly using only the supplied tool observations when they are present. Include PDF page citations when the observations contain page numbers.",
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nTool observations:\n{observation_text}",
                },
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content or observation_text
    except Exception:
        return observation_text


def direct_answer(question: str) -> str:
    try:
        completion = get_groq_client().chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": "Answer directly and concisely without using tools."},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content or "I could not produce an answer."
    except Exception as exc:
        return f"I could not produce a direct answer: {exc}"


def run_recovery_agent(question: str) -> str:
    observations: list[dict[str, str]] = []
    lower = question.lower()

    if "wikipedia" in lower:
        result = execute_tool("get_wikipedia_summary", {"topic": _extract_wikipedia_topic(question)})
        observations.append({"tool": "get_wikipedia_summary", "result": result})
        return synthesize_answer_from_observations(question, observations)

    if _looks_like_domain_kb(question):
        result = execute_tool("search_knowledge_base", {"query": question})
        observations.append({"tool": "search_knowledge_base", "result": result})

    if _looks_like_current_web(question):
        result = execute_tool("search_web", {"query": question})
        observations.append({"tool": "search_web", "result": result})

    if _looks_like_math(question):
        result = execute_tool("calculate", {"expression": _extract_math_expression(question)})
        observations.append({"tool": "calculate", "result": result})

    if observations:
        return synthesize_answer_from_observations(question, observations)

    return direct_answer(question)


def run_agent(question: str) -> str:
    client = get_groq_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = build_tool_schemas()

    for _ in range(settings.max_agent_iterations):
        try:
            completion = client.chat.completions.create(
                model=settings.llm_model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception:
            return run_recovery_agent(question)
        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls or []

        if not tool_calls:
            return response_message.content or "I could not produce an answer."

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [_tool_call_to_payload(tool_call) for tool_call in tool_calls],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = _parse_tool_args(tool_name, tool_call.function.arguments or "{}")
            tool_result = execute_tool(tool_name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result[:8000],
                }
            )

    return "The agent reached its maximum tool-iteration limit before producing a final answer."


def get_vector_count() -> int:
    try:
        stats = _to_plain_dict(get_pinecone_index().describe_index_stats())
        namespaces = stats.get("namespaces", {}) or {}
        namespace_stats = namespaces.get(settings.pinecone_namespace, {}) or {}
        return int(namespace_stats.get("vector_count", 0) or 0)
    except Exception:
        return 0


@app.get("/")
def serve_chat() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
def api_root() -> dict[str, Any]:
    return {
        "app": "Agentic RAG Lab 06",
        "owner": "M Abdullah Fawad",
        "endpoints": ["/health", "/tools", "/ingest", "/query", "/logs", "/docs"],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "owner": "M Abdullah Fawad",
        "pinecone_vectors": get_vector_count(),
        "pinecone_namespace": settings.pinecone_namespace,
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
        "pdf_configured": settings.domain_pdf_path.exists(),
        "web_search_enabled": bool(settings.tavily_api_key),
        "groq_configured": bool(settings.groq_api_key),
        "pinecone_configured": bool(settings.pinecone_api_key and (settings.pinecone_host or settings.pinecone_index_name)),
    }


@app.get("/tools")
def tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "description": (fn.__doc__ or "").strip(),
                "parameters": TOOL_PARAMETERS[name],
            }
            for name, fn in TOOL_FUNCTIONS.items()
        ]
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_pdf() -> IngestResponse:
    pdf_path = settings.domain_pdf_path
    if not pdf_path.exists():
        fallback = BASE_DIR / "ai_agents.pdf"
        if fallback.exists():
            pdf_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"PDF not found at {settings.domain_pdf_path}")

    try:
        extracted_pages = extract_pdf_pages(pdf_path)
        all_chunks: list[dict[str, Any]] = []
        pages_indexed = 0
        for page_data in extracted_pages:
            chunks = chunk_page_text(page_data["text"], int(page_data["page"]))
            if chunks:
                pages_indexed += 1
                all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable text was found in the PDF and no sidecar text cache exists.",
            )

        index = get_pinecone_index()
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embed_texts(texts)

        batch_size = 50
        for start in range(0, len(all_chunks), batch_size):
            vectors = []
            for chunk, embedding in zip(all_chunks[start : start + batch_size], embeddings[start : start + batch_size]):
                vector_id = (
                    f"{settings.pinecone_namespace}-"
                    f"page-{chunk['page']:03d}-chunk-{chunk['chunk_index']:03d}"
                )
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "text": chunk["text"],
                            "page": chunk["page"],
                            "chunk_index": chunk["chunk_index"],
                            "source": pdf_path.name,
                            "namespace": settings.pinecone_namespace,
                        },
                    }
                )
            index.upsert(vectors=vectors, namespace=settings.pinecone_namespace)

        return IngestResponse(
            status="success",
            chunks_ingested=len(all_chunks),
            pages_indexed=pages_indexed,
            namespace=settings.pinecone_namespace,
            source_file=pdf_path.name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest) -> QueryResponse:
    trace: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    trace_token = _tool_trace.set(trace)
    citation_token = _citations.set(citations)
    try:
        answer = run_agent(request.question)
        tools_used = list(dict.fromkeys(item["tool"] for item in trace))
        return QueryResponse(
            answer=answer,
            tools_used=tools_used,
            tool_trace=trace,
            citations=citations,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent query failed: {exc}") from exc
    finally:
        _tool_trace.reset(trace_token)
        _citations.reset(citation_token)


@app.get("/logs")
def logs(limit: int = Query(50, ge=1, le=300)) -> dict[str, Any]:
    if not LOG_PATH.exists():
        return {"path": str(LOG_PATH.name), "lines": []}
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    return {"path": str(LOG_PATH.name), "lines": lines[-limit:]}
