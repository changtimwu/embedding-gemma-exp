import sys, time, json, urllib.request
URL = "http://127.0.0.1:8080/v1/embeddings"

def embed(**body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req))
    return [d["embedding"] for d in r["data"]], r["usage"], time.perf_counter() - t

dot = lambda a, b: sum(x * y for x, y in zip(a, b))
docs = ["Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet."]

q, _, _ = embed(input="Which planet is known as the Red Planet?", prompt="task: search result | query: ")
for dim in (768, 256, 128):
    q, _, _ = embed(input="Which planet is known as the Red Planet?", prompt="task: search result | query: ", dimensions=dim)
    d, usage, dt = embed(input=docs, prompt="title: none | text: ", dimensions=dim)
    print(f"dim={dim:3d} venus={dot(q[0], d[0]):.4f} mars={dot(q[0], d[1]):.4f} ({dt*1000:.0f} ms)")

img = sys.argv[1] if len(sys.argv) > 1 else "samples/dog.jpg"
v, usage, dt = embed(input=["A photo of a dog", "A photo of a car",
                            [{"type": "image", "url": img}]])
print(f"image: dog-text={dot(v[2], v[0]):.4f} car-text={dot(v[2], v[1]):.4f} ({dt*1000:.0f} ms, {usage})")
