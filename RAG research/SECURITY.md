# Security Policy

## Reporting a vulnerability
- Email: **javiercastro@aiready.es**.
- Include a clear description, reproduction steps, and any logs. Do **not** include secrets or personal data.

## Supported versions
- Main branch (`streamlit_rag.py`, `rag_downloads.py`) as committed in this repository. No older releases are maintained.

## Handling secrets
- Keep `OPENAI_API_KEY` in your shell environment or a local `.env` that is **not** committed.
- Never paste keys into prompts or notebooks that may be shared.
- Avoid committing `.rag_downloads.index.pkl` if it may contain sensitive document paths.

## Hardening guidance
- Run in a least-privilege environment; restrict access to the `~/Downloads/rag` directory if it has sensitive files.
- Prefer per-user API keys with minimal scopes.
- Review your `rag_downloads.py` args before sharing outputs to ensure no sensitive file paths are leaked.
- Clear chat history with the “Clear chat history” button if others use the same machine.

## Dependencies
- Python packages: `sentence-transformers`, `pypdf`, `openai`, `streamlit`, `ragas`, `datasets`, `langchain-openai`, `numpy`.
- Update regularly: `python3 -m pip install --upgrade <package>`.
- If you vendor new dependencies, pin versions and review changelogs for security fixes.

## Data privacy
- Documents remain local; only prompts/contexts may be sent to OpenAI if `OPENAI_API_KEY` is set.
- If privacy is critical, run with no API key; the app will use extractive summaries instead of LLM calls.
