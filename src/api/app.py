from flask import Flask, request, jsonify
from config.config import Config
import numpy as np
from src.models.lstm_model import LSTMModel

app = Flask(__name__)

# Load model at startup
model = LSTMModel.load(f"{Config.MODEL_DIR}/lstm_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_sequence = np.array(data["sequence"]).reshape(1, -1, 1)
        prediction = model.predict(input_sequence)
        return jsonify({
            "status": "success",
            "prediction": float(prediction[0][0])
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host=Config.API_HOST, port=Config.API_PORT)
