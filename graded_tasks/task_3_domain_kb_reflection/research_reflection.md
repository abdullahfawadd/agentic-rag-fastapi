# Research Reflection: Agentic RAG for AI Agents

**Student:** M Abdullah Fawad  
**Knowledge base:** `data/ai_agents.pdf`  
**Pinecone namespace:** `ai_agents_pdf`

## What Worked Well

The knowledge-base tool works best for questions that are clearly grounded in the uploaded AI agents PDF, such as questions about tool calling, agentic RAG, reasoning loops, planning, and the difference between deterministic retrieval and autonomous tool selection. These questions work well because the query can be embedded, matched to semantically similar PDF chunks, and returned with page citations that make the answer easy to verify.

## Tool Selection Issues

The agent can choose the wrong tool when the question is broad or ambiguous. For example, a question like “What are the best AI agent frameworks?” may be interpreted as a web-search task even if the student wanted concepts from the PDF. I would improve this by making the knowledge-base tool docstring more explicit: use it for questions about the uploaded AI agents PDF, course material, lab theory, agent architecture, tool calling, ReAct, and RAG, but avoid it for recent releases or current industry comparisons.

## Production Deployment

For production, I would deploy the FastAPI backend behind HTTPS with secrets managed by environment variables or a cloud secret manager. Pinecone would store domain-specific namespaces per course, document set, or user. The frontend would show citations and tool traces by default so users can audit the agent’s behavior. I would add authentication, rate limits, request logging, automated tests, and monitoring for failed tool calls, latency, vector counts, and model errors. For larger document collections, ingestion should run as a background job with progress tracking instead of a blocking request.
