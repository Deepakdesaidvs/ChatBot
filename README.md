# Local Document AI

Streamlit document chat with PDF/OCR extraction, citations, and either local Ollama
or hosted OpenAI chat and embeddings.

## Deploy publicly with OpenAI on Render

This app's OpenAI mode sends extracted document text, questions, and embedding
inputs to the OpenAI API. Do not upload confidential documents unless that data
transfer is acceptable for your use case.

1. Push this project to a private or public GitHub repository. Do not include
   `.env`, `.venv/`, or `data/`; these are excluded by `.gitignore`.
2. In Render, create a **Web Service** from the repository.
3. Set the build command to `pip install -r requirements.txt`.
4. Set the start command to
   `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.
5. Add these environment variables in Render's dashboard. Use Python `3.11.9`
   to match the tested local runtime.

   | Name | Value |
   | --- | --- |
   | `PYTHON_VERSION` | `3.11.9` |
   | `LLM_PROVIDER` | `openai` |
   | `OPENAI_API_KEY` | Your OpenAI API key (mark it secret) |
   | `OPENAI_CHAT_MODEL` | `gpt-4o-mini` |
   | `OPENAI_EMBED_MODEL` | `text-embedding-3-small` |
   | `APP_PASSWORD` | A long, unique password to protect the shared app |

6. Deploy the service and open its HTTPS URL. Share the URL and application
   password only with people you intend to allow. Set usage limits/alerts in
   your OpenAI account; the shared password is not per-user billing or a
   request quota.

The app uses in-memory Streamlit sessions and deliberately deletes old
collections on process startup. A server restart therefore removes uploaded
documents and conversation state. Users must re-upload their files after a
restart. Avoid multiple web-service instances because each instance has its
own in-memory session state.

For local use, leave `LLM_PROVIDER=ollama` and configure Ollama as before. Never
put an API key in source control or in a browser/client-side setting.