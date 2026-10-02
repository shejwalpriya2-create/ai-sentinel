from flask import Flask, render_template, request, jsonify
from model import detect_spam

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please enter a message"
        }), 400

    result, confidence = detect_spam(message)

    return jsonify({
        "result": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)