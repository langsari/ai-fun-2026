from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__, template_folder="templates")
COLAB_URL = ""

@app.route("/")
def home(): return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    d = request.get_json()
    q = d.get("question","").strip()
    if not q: return jsonify({"error":"Empty question"}), 400
    try:
        r = requests.post(f"{COLAB_URL}/ask",
                          json={"question":q,"top_k":d.get("top_k",3)}, timeout=60)
        return jsonify(r.json()) if r.ok else jsonify({"error":f"Colab {r.status_code}"}), 500
    except requests.ConnectionError:
        return jsonify({"error":"Cannot reach Colab — is Cell B running?"}), 500
    except requests.Timeout:
        return jsonify({"error":"Colab timeout"}), 500

@app.route("/health")
def health():
    try:
        r = requests.get(f"{COLAB_URL}/health", timeout=5)
        return jsonify({"status":"connected","colab":r.json()}) if r.ok                else jsonify({"status":"disconnected"}), 500
    except: return jsonify({"status":"disconnected"}), 500

@app.route("/update_url", methods=["POST"])
def update_url():
    global COLAB_URL
    COLAB_URL = request.get_json().get("url","").strip().rstrip("/")
    return jsonify({"status":"ok","url":COLAB_URL})

if __name__ == "__main__":
    print(f"UI → http://localhost:8080  |  Colab: {COLAB_URL}")
    app.run(host="0.0.0.0", port=8080, debug=True)
