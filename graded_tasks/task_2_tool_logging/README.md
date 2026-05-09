# Graded Task 2: Tool Call Logging

## What to Demonstrate

- Every tool writes to `tool_log.txt` at runtime.
- Each line includes timestamp, tool name, input query/expression/topic, and success/failure.
- After at least five tool-using queries, click **Logs** in the UI or open `tool_log.txt`.

## Screenshot Checklist

- Screenshot 1: Chat UI logs panel showing at least five entries.
- Screenshot 2: Local `tool_log.txt` opened in the editor or terminal.

## Example Format

```text
2026-03-12 14:32:01 | search_knowledge_base | query='AI agents tool use' | success
2026-03-12 14:33:20 | get_wikipedia_summary | query='Transformer architecture' | success
```
