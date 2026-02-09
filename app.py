"""FastAPI web application for the paper-to-notebook tool."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

from config import MAX_PDF_SIZE_MB, DEFAULT_MODEL
from web_pipeline import run_web_pipeline

# --- Configuration ---
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", str(MAX_PDF_SIZE_MB)))

# --- App setup ---
app = FastAPI(title="Paper to Notebook", version="1.3", docs_url=None, redoc_url=None)

# Temp directory for generated notebooks
TEMP_DIR = tempfile.mkdtemp(prefix="paper2nb_")

# Concurrency limiter
_generation_semaphore = asyncio.Semaphore(3)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/api/generate")
async def generate(request: Request, file: UploadFile = File(...), api_key: str = Form(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    api_key = api_key.strip()
    if not api_key:
        raise HTTPException(400, "OpenAI API key is required")

    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

    job_id = uuid.uuid4().hex[:12]
    draft_id = job_id + "_draft"

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_progress(step: int, name: str, detail: str, extra: dict = None):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("progress", step, name, detail, extra)),
                loop,
            )

        def on_thinking(text: str):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("thinking", text)),
                loop,
            )

        async def run_in_thread():
            async with _generation_semaphore:
                return await loop.run_in_executor(
                    None,
                    lambda: run_web_pipeline(
                        pdf_bytes, DEFAULT_MODEL, on_progress,
                        api_key=api_key, on_thinking=on_thinking,
                    ),
                )

        task = asyncio.create_task(run_in_thread())

        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)

                # Handle thinking events
                if event[0] == "thinking":
                    data = json.dumps({"text": event[1]})
                    yield f"event: thinking\ndata: {data}\n\n"
                    continue

                _, step, name, detail, extra = event

                # Check if this progress event carries draft notebook bytes
                if extra and "draft_bytes" in extra:
                    draft_bytes = extra.pop("draft_bytes")
                    # Save draft to disk
                    draft_path = os.path.join(TEMP_DIR, f"{draft_id}.ipynb")
                    with open(draft_path, "wb") as f:
                        f.write(draft_bytes)
                    # Send progress event (without the bytes)
                    data = {"step": step, "name": name, "detail": detail, "extra": extra}
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                    # Send draft_ready event
                    draft_data = json.dumps({"job_id": draft_id, "size_kb": len(draft_bytes) // 1024})
                    yield f"event: draft_ready\ndata: {draft_data}\n\n"
                else:
                    data = {"step": step, "name": name, "detail": detail}
                    if extra:
                        data["extra"] = extra
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

        # Drain remaining
        while not progress_queue.empty():
            event = await progress_queue.get()
            if event[0] == "thinking":
                continue
            _, step, name, detail, extra = event
            if extra and "draft_bytes" in extra:
                extra.pop("draft_bytes")
            data = {"step": step, "name": name, "detail": detail}
            if extra:
                data["extra"] = extra
            yield f"event: progress\ndata: {json.dumps(data)}\n\n"

        try:
            notebook_bytes = task.result()
            output_path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
            with open(output_path, "wb") as f:
                f.write(notebook_bytes)
            data = json.dumps({"job_id": job_id, "size_kb": len(notebook_bytes) // 1024})
            yield f"event: complete\ndata: {data}\n\n"
        except Exception as e:
            data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "Invalid job ID")
    path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
    if not os.path.exists(path):
        raise HTTPException(404, "Notebook not found or expired")
    return FileResponse(
        path,
        media_type="application/x-ipynb+json",
        filename="generated_notebook.ipynb",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
