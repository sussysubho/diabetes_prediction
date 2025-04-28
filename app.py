# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for frontend integration (like Firebase)

# Define the Model Architecture (exactly same as training)
class DiabetesModel(nn.Module):
    def __init__(self, input_dim, scaler_y=None):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(16, 1)
        self.scaler_y = scaler_y  # Attach scaler inside model (optional)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)
        return self.fc3(x)

# Load Model
device = torch.device('cpu')  # Always CPU on Render

# Initialize empty model
model = DiabetesModel(input_dim=10)

# Load checkpoint
checkpoint = torch.load("Dark-knight_model.pkl", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.scaler_y = checkpoint.get("scaler_y", None)

model.to(device)
model.eval()

# API Routes

@app.route("/")
def home():
    return "🔥 Diabetes Prediction API is live on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[data["age"], data["sex"], data["bmi"], data["bp"],
                              data["s1"], data["s2"], data["s3"], data["s4"],
                              data["s5"], data["s6"]]], dtype=np.float32)

        input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        with torch.no_grad():
            scaled_prediction = model(input_tensor).cpu().numpy()[0][0]

        # If scaler_y exists, unscale the prediction
        if model.scaler_y:
            unscaled_prediction = model.scaler_y.inverse_transform(np.array([[scaled_prediction]]))[0][0]
        else:
            unscaled_prediction = scaled_prediction

        return jsonify({
            "scaled_prediction": float(scaled_prediction),
            "unscaled_prediction": float(unscaled_prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entry point for Render deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
