from flask import Flask, request, Response, render_template
import requests
import json

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    # Use model selected from frontend (falls back to mistral)
    model = data.get("model", "mistral")

    history = data.get("history", [])
    system = data.get("system", "You are AbhijitAI, a helpful AI assistant.")

    # Keep last 20 messages for better context
    trimmed_history = history[-20:]

    # Build prompt
    prompt_parts = [f"[SYSTEM]: {system}\n"]

    for msg in trimmed_history:
        role = "User" if msg["role"] == "user" else "AbhijitAI"
        prompt_parts.append(f"{role}: {msg['content']}")

    prompt_parts.append(f"User: {user_input}")
    prompt_parts.append("AbhijitAI:")

    full_prompt = "\n".join(prompt_parts)

    # Stream response
    def generate():
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 4096,
                        "num_predict": -1       # -1 = no limit, full response always
                    },
                    "keep_alive": "10m"
                },
                stream=True,
                timeout=300                     # 5 min timeout for long responses
            )

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

        except requests.exceptions.ConnectionError:
            yield "\n\nCannot connect to Ollama. Make sure it is running on port 11434."
        except requests.exceptions.Timeout:
            yield "\n\nRequest timed out. The model may be slow or busy — try again."
        except Exception as e:
            yield f"\n\nError: {str(e)}"

    return Response(generate(), content_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(debug=True, port=5000)