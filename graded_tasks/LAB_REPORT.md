# Lab 06 Report: Agentic RAG with FastAPI, Groq, Pinecone, Tavily, and Wikipedia

**Student Name:** M Abdullah Fawad  
**Repository:** `https://github.com/abdullahfawadd/agentic-rag-fastapi`  
**Domain Knowledge Base:** `data/ai_agents.pdf`  
**Pinecone Namespace:** `ai_agents_pdf`  
**LLM:** Groq `llama-3.3-70b-versatile`  
**Embedding Model:** `BAAI/bge-small-en-v1.5`  

## 1. Objective

This lab extends a deterministic RAG pipeline into an agentic RAG assistant. The system uses a FastAPI backend and a browser UI where the agent can decide whether to call the Pinecone knowledge-base search, Tavily live web search, a safe calculator, or a Wikipedia summary tool.

**Screenshot Placeholder 1: Final UI Home**

Paste a screenshot of the app home page here.

Capture:
- URL bar showing `http://127.0.0.1:8000`
- Sidebar with `M Abdullah Fawad`
- Home screen prompt cards
- Health/vector status visible

## 2. System Architecture

The app uses FastAPI for REST endpoints, Groq for LLM tool calling, Pinecone for vector search, Tavily for live web search, Wikipedia REST API for encyclopedia summaries, and a modern HTML/CSS/JavaScript UI for interaction and screenshots.

**Screenshot Placeholder 2: Swagger API Docs**

Paste a screenshot of `/docs` here.

Capture:
- URL `http://127.0.0.1:8000/docs`
- Endpoints visible: `/health`, `/tools`, `/ingest`, `/query`, `/logs`
- Do not show `.env` or API keys

## 3. Environment and Tool Setup

The application is configured through `.env`, while `.env.example` documents the required variables without exposing secrets.

Tools/APIs used:
- **Groq:** LLM reasoning and tool-calling
- **Pinecone:** Vector database for PDF chunks
- **Tavily:** Live web search
- **Wikipedia REST API:** Article summaries
- **FastAPI:** Backend API
- **Sentence Transformers:** Local embeddings

**Screenshot Placeholder 3: Pinecone Index**

Paste a Pinecone dashboard screenshot here.

Capture:
- Index name: `lab06-agentic-rag`
- Host matching the configured Pinecone host
- Dimension should be `384` for `BAAI/bge-small-en-v1.5`
- Namespace/vector count if visible
- Do not show full API keys

**Screenshot Placeholder 4: Health Endpoint**

Paste a screenshot of `/health` or the UI Health result here.

Capture expected values:
- `status: ok`
- `pinecone_vectors` greater than `0`
- `pinecone_namespace: ai_agents_pdf`
- `web_search_enabled: true`
- `groq_configured: true`
- `pinecone_configured: true`

## 4. Base Lab Tool Registration

The `/tools` endpoint confirms that all agent tools are registered and visible to the agent.

**Screenshot Placeholder 5: Tools Endpoint**

Paste a screenshot of `http://127.0.0.1:8000/tools` here.

Capture all tools:
- `search_knowledge_base`
- `search_web`
- `calculate`
- `get_wikipedia_summary`

## 5. Graded Task 1: Wikipedia Summary Tool

### Implementation Summary

A fourth tool named `get_wikipedia_summary(topic: str)` was added. It calls the Wikipedia REST API and returns the article title, opening summary, and article URL. The tool docstring tells the agent to use it for stable encyclopedia-style questions and not for live news, PDF content, or calculations.

### Demo Prompt

Use this exact prompt:

```text
Give me a Wikipedia summary of the Transformer architecture in NLP.
```

Expected result:
- `tools_used` contains `get_wikipedia_summary`
- Tool trace shows input topic similar to `Transformer architecture`
- Answer contains a Wikipedia-style summary

**Screenshot Placeholder 6: Wikipedia Tool Trace**

Paste the chat UI screenshot here.

Capture:
- User prompt visible
- Assistant answer visible
- Expanded Agent Tool Trace visible
- Tool name `get_wikipedia_summary` visible

## 6. Graded Task 2: Tool Call Logging

### Implementation Summary

Every tool appends a line to `tool_log.txt` with timestamp, tool name, input, and success/failure. The UI also exposes recent logs through the **Tool logs** button.

### Five Queries to Run Before Screenshot

Run these prompts so the log contains at least five entries:

```text
Give me a Wikipedia summary of the Transformer architecture in NLP.
```

```text
What is 2 raised to the power of 16?
```

```text
What does the AI agents PDF say about tool design?
```

```text
What are the latest AI agent frameworks?
```

```text
What does the PDF say about tool use, and how many days are in 5 years?
```

Expected log format:

```text
2026-05-10 02:47:43 | search_knowledge_base | query='What does the AI agents PDF say about tool design?' | success
```

**Screenshot Placeholder 7: Logs in UI**

Paste a screenshot after clicking **Tool logs**.

Capture:
- At least five log lines
- Tool names visible
- Success status visible

**Screenshot Placeholder 8: tool_log.txt**

Paste a screenshot of `tool_log.txt` opened in VS Code or terminal.

Capture:
- Timestamp
- Tool name
- Query/input
- `success` status

## 7. Graded Task 3: Domain Knowledge Base and Research Reflection

### Implementation Summary

The default Paul Graham PDF was replaced with the AI agents domain PDF: `data/ai_agents.pdf`. The Pinecone namespace is `ai_agents_pdf`. Because this PDF is image-only, the app includes `data/ai_agents_ocr.txt` as the sidecar text source for ingestion while preserving page-style citations.

### Ingestion Evidence

Run:

```text
POST http://127.0.0.1:8000/ingest
```

Expected:
- `status: success`
- `chunks_ingested` greater than `0`
- `pages_indexed` greater than `0`
- `namespace: ai_agents_pdf`

**Screenshot Placeholder 9: Ingest Success**

Paste `/ingest` response or UI ingest success here.

Capture:
- Chunks ingested
- Pages indexed
- Namespace

**Screenshot Placeholder 10: Pinecone Vector Count After Ingest**

Paste Pinecone dashboard or `/health` after ingest.

Capture:
- Vector count greater than `0`
- Namespace `ai_agents_pdf`

### Five Domain Questions

Use these prompts for Task 3 screenshots:

```text
What does the AI agents PDF say about tool design?
```

```text
How does the PDF explain agentic RAG?
```

```text
What does the AI agents PDF say about live web search?
```

```text
Why is tool-call logging important in a production agentic RAG system?
```

```text
What does the PDF say about safe calculator tools?
```

At least three answers should:
- Use `search_knowledge_base`
- Show page citations
- Show expanded tool trace

**Screenshot Placeholder 11: Domain Question 1**

Paste screenshot for:

```text
What does the AI agents PDF say about tool design?
```

Capture citations and tool trace.

**Screenshot Placeholder 12: Domain Question 2**

Paste screenshot for:

```text
How does the PDF explain agentic RAG?
```

Capture citations and tool trace.

**Screenshot Placeholder 13: Domain Question 3**

Paste screenshot for:

```text
What does the AI agents PDF say about live web search?
```

Capture citations and tool trace.

**Screenshot Placeholder 14: Domain Question 4**

Paste screenshot for:

```text
Why is tool-call logging important in a production agentic RAG system?
```

Capture citations and tool trace.

**Screenshot Placeholder 15: Domain Question 5**

Paste screenshot for:

```text
What does the PDF say about safe calculator tools?
```

Capture citations and tool trace.

## 8. Research Reflection

### What Questions Worked Well with the KB Tool?

Questions about agentic RAG, tool design, vector search, tool logging, safe calculators, and production deployment worked well because they matched content stored in the AI agents knowledge base. These questions returned relevant chunks and page citations.

### Which Questions Could Cause Wrong Tool Selection?

Broad questions like “What are the best AI agent frameworks?” may cause the agent to use both the knowledge base and web search. This is acceptable for mixed questions, but if the goal is only current frameworks, the tool docstring should more strongly prefer web search for recent information.

### How Would This Be Deployed in Production?

In production, this agent would run behind HTTPS with authentication, rate limits, monitoring, structured logs, background ingestion jobs, secret management, and separate Pinecone namespaces per domain or user. The UI should keep citations and tool traces visible to increase trust.

## 9. Final Full-Marks Checklist

- `/tools` shows all four tools
- `/health` shows configured Groq, Pinecone, Tavily, and vector count
- Pinecone dashboard screenshot included
- Wikipedia tool screenshot included
- Tool log screenshot with at least five entries included
- `/ingest` success screenshot included
- Five domain KB screenshots included
- At least three domain answers show citations
- Final UI screenshot included
- `.env` and API keys are not shown in screenshots
- GitHub repository uses `abdullahfawadd <abdullahfawad.dev@gmail.com>`
