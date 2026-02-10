import base64

filename = "architecture.mmd"

with open(filename, "r") as f:
    graph = f.read()

graph_bytes = graph.encode("ascii")
base64_bytes = base64.b64encode(graph_bytes)
base64_string = base64_bytes.decode("ascii")

url = f"https://mermaid.ink/img/{base64_string}"

print(f"Generated URL for {filename}: {url}")
