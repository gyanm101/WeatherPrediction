from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import MeanAbsoluteError

custom_objects = {"mae": MeanAbsoluteError()}

lstm_model = tf.keras.models.load_model("lstm_model.h5", custom_objects=custom_objects)

ridge_model = joblib.load("ridge_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_type = data.get("model", "ridge")
    features = np.array(data["features"]).reshape(1, -1)

    expected_features = 23 

    if features.shape[1] != expected_features:
        return jsonify({
            "error": f"Expected {expected_features} features, but got {features.shape[1]}."
        }), 400

    if model_type == "ridge":
        prediction = ridge_model.predict(features)[0]
    elif model_type == "xgb":
        prediction = xgb_model.predict(features)[0]
    elif model_type == "lstm":
        features = features.reshape(1, features.shape[1], 1)
        prediction = lstm_model.predict(features)[0][0]
    else:
        return jsonify({"error": "Invalid model type"}), 400

    return jsonify({"model": model_type, "prediction": float(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
