import json, os
from flask import Flask, jsonify, render_template

app = Flask(__name__)
RESULTS = os.path.join(os.path.dirname(__file__), "data", "results.json")

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/results")
def results():
    with open(RESULTS) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    if not os.path.exists(RESULTS):
        raise SystemExit("Run `python precompute.py` first to create data/results.json")
    app.run(debug=False, port=5000)
