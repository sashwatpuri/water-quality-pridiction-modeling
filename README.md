# 💧 IoT Water Quality Management System

A comprehensive system for monitoring and predicting water quality using IoT data integration and Machine Learning. This project combines a Flask backend with an interactive frontend to analyze water parameters and determine potability.

## 🚀 Features

- **Real-time Prediction**: Uses an optimized XGBoost Machine Learning model to predict if water is potable.
- **IoT Integration**: Fetches real-time water quality data from ThingSpeak channels.
- **Interactive Dashboard**: User-friendly interface for monitoring parameters and viewing results.
- **Health Risk Assessment**: Provides detailed health risk warnings for unsafe parameter levels.
- **User System**: Secure login and registration with personalized settings for IoT keys.
- **Visualizations**: Dynamic charts to track water quality trends.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Chart.js
- **IoT Platform**: ThingSpeak API

## 📂 Project Structure

```
├── app.py                      # Flask backend API
├── train_model_optimized.py    # Script to train the XGBoost model
├── index.html                  # Main frontend application
├── water_potability_cleaned.csv # Dataset for training
├── Optimized_Water_XGBoost_Model.pkl # Trained ML model
└── README.md                   # Project documentation
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
Create a `requirements.txt` or install directly:
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost joblib
```

### 3. Train the Model (Optional)
If you want to retrain the model:
```bash
python train_model_optimized.py
```
This will generate `Optimized_Water_XGBoost_Model.pkl`.

### 4. Run the Backend Server
Start the Flask API:
```bash
python app.py
```
The server will start at `http://localhost:5000`.

### 5. Launch the Frontend
Simply open `index.html` in your web browser.
*Note: For best performance and to avoid CORS issues with some browsers, it's recommended to serve the HTML file using a simple HTTP server or VS Code Live Server.*

## 🖥️ Usage Guide

1.  **Register/Login**: Create an account. You can optionally save your ThingSpeak API Key and Channel ID during registration.
2.  **Dashboard**:
    *   **IoT Data**: Click "Load IoT Data" to fetch the latest readings from your ThingSpeak channel.
    *   **Manual Input**: Enter pH, Hardness, Organic Carbon, Conductivity, and Turbidity manually.
    *   **Random Values**: Use the "Load Random Values" button for testing.
3.  **Analyze**: Click "Analyze Water Quality" to get a prediction (Safe/Unsafe) and a detailed health risk report.

## 📊 API Endpoints

### `POST /api/predict`
Predicts water potability.
**Body:**
```json
{
    "ph": 7.5,
    "hardness": 200,
    "organic_carbon": 15,
    "conductivity": 400,
    "turbidity": 3
}
```

### `GET /api/status`
Checks if the API and model are running.

### `GET /api/model-info`
Returns information about the loaded ML model.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License
This project is open-source and available under the MIT License.
