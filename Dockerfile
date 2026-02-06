FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch and Jupyter kernel
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.web.txt .
RUN pip install --no-cache-dir -r requirements.web.txt

# Install IPython kernel for notebook execution
RUN python -m ipykernel install --user --name python3

COPY config.py llm.py pipeline.py prompts.py notebook_builder.py \
     generate_notebook.py web_pipeline.py app.py ./
COPY static/ static/

ENV PORT=8000
ENV GOOGLE_API_KEY=AIzaSyDm1W7c6Nm6b8OTa1QNsl8gQm5wyK3A5_c

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
