import base64

graph = """graph TD
    User(User) -->|Upload PDF| API[FastAPI Server]
    API -->|Start Job| Pipeline{Pipeline Orchestrator}
    
    Pipeline -->|1. Analyze| Analysis[Paper Analysis]
    Analysis -->|2. Plan| Design[Toy Implementation Design]
    Design -->|3. Code| Gen[Code Generation]
    Gen -->|4. Refine| Validate[Validation & Repair]
    
    Validate -->|Save| NB[(Generated .ipynb)]
    NB -->|Download| User"""

graph_bytes = graph.encode("ascii")
base64_bytes = base64.b64encode(graph_bytes)
base64_string = base64_bytes.decode("ascii")

url = f"https://mermaid.ink/img/{base64_string}"

print(f"Generated URL: {url}")
