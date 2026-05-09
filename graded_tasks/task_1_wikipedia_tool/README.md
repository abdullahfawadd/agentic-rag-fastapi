# Graded Task 1: Wikipedia Summary Tool

## What to Demonstrate

- `GET /tools` includes `get_wikipedia_summary`.
- A query like `Give me a Wikipedia summary of the Transformer architecture in NLP.` uses the Wikipedia tool.
- The UI shows `get_wikipedia_summary` in the tool trace.

## Screenshot Checklist

- Screenshot 1: `/tools` response showing all four tools.
- Screenshot 2: Chat UI answer with expanded tool trace for the Wikipedia prompt.

## Expected Result

The agent should call `get_wikipedia_summary`, not `search_web`, for a stable encyclopedia-style Wikipedia request.
