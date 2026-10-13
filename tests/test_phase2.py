"""Phase 2 check: settings load, logging works, Ollama is reachable."""
from src.config import settings
from src.utils import check_ollama, human_size, new_session_id, session_dir, setup_logging

setup_logging()

print("LLM model      :", settings.llm_model)
print("Embedding model:", settings.embed_model)
print("Ollama host    :", settings.ollama_host)
print("Chunk size     :", settings.chunk_size, "/ overlap", settings.chunk_overlap)
print("Max file size  :", human_size(settings.max_file_bytes))

sid = new_session_id()
folder = session_dir(sid)
print("Session folder :", folder.exists(), "-", folder.name[:8] + "...")
folder.rmdir()  # clean up the test folder

status = check_ollama()
print("Ollama running :", status["running"])
print("Installed      :", status["installed"])
print("Missing models :", status["missing"] or "none")
if status["error"]:
    print("Error          :", status["error"])