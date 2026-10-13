# Local Document AI

Streamlit document chat with PDF/OCR extraction, citations, and either local Ollama
or hosted OpenAI chat and embeddings.

## Public deployment with OpenAI on Render

This app's OpenAI mode sends extracted document text, questions, and embedding
inputs to the OpenAI API. Do not upload confidential documents unless that data
transfer is acceptable for your use case.

The root `render.yaml` is a Render Blueprint. To deploy:

1. Push the project to GitHub. `.env`, `.venv/`, and `data/` are excluded by
   `.gitignore`; never commit an API key.
2. In Render, create a **Blueprint** and connect the GitHub repository.
3. Enter your OpenAI API key when Render prompts for `OPENAI_API_KEY`. Keep it
   in Render's secret environment settings, never in source control.
4. Apply the Blueprint and wait for the first deploy. Render provides the
   public HTTPS URL for the service; anyone with that link can open and use the
   chatbot without an application password.

This public configuration deliberately leaves `APP_PASSWORD` unset. Anyone
with the link can use the same service and its OpenAI account, so set OpenAI
usage limits/alerts before sharing the URL. The app does not currently enforce
`RATE_LIMIT_PER_MINUTE` as a per-user API quota.

The app uses in-memory Streamlit sessions and deliberately deletes old
collections on process startup. A server restart therefore removes uploaded
documents and conversation state. On Render's free service, inactivity can also
put the app to sleep; users may need to wait for it to wake up, and files must
be uploaded again after a restart. Avoid multiple web-service instances because
each instance has its own in-memory session state.

For local use, leave `LLM_PROVIDER=ollama` and configure Ollama as before. Never
put an API key in source control or in a browser/client-side setting.