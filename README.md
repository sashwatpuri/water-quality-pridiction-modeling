# 💧 AquaSense - IoT Water Quality Prediction System

A comprehensive AI-powered system for monitoring and predicting water potability using Machine Learning and IoT integration. This project combines a Flask backend API with an interactive web dashboard to analyze 9 water quality parameters and provide detailed health risk assessments.

## 🚀 Features

- **Multi-Parameter Analysis**: Analyzes 9 critical water quality parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity)
- **Machine Learning Model**: Random Forest classifier trained on comprehensive water quality dataset
- **Real-time Predictions**: Instant potability predictions with confidence scores
- **Detailed Risk Assessment**: Parameter-specific health risks and recommended actions for each value
- **IoT Integration**: Fetches real-time data from ThingSpeak IoT channels
- **Interactive Dashboard**: User-friendly web interface with parameter input and visual analysis
- **Actionable Recommendations**: Specific, numbered action steps to improve water quality
- **Data Visualization**: Dynamic charts tracking water quality trends over time
- **Export Functionality**: Download analysis reports as CSV files

## 🛠️ Tech Stack

- **Backend**: Python 3.x, Flask, Flask-CORS
- **Machine Learning**: Scikit-learn (Random Forest), Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Chart.js
- **IoT Platform**: ThingSpeak API
- **Notebook**: Jupyter Notebook (Model.ipynb)

## 📂 Project Structure

```
ET201/
├── app.py                           # Flask backend API server
├── dashboard.html                   # Main interactive dashboard (in Websites/)
├── Model.ipynb                      # Jupyter notebook for model exploration
├── water_potability.csv             # Dataset for model training
├── README.md                        # Project documentation
│
├── Models/
│   └── RandomForest_Model.pkl       # Trained Random Forest model
│
├── Training_Models/
│   ├── train_model.py              # Initial model training script
│   ├── train_model2.py             # Optimized training version
│   └── train_model3.py             # Final model training version
│
└── Websites/
    ├── dashboard.html              # Main water analyzer dashboard
    ├── index.html                  # Login/registration page
    └── website.html                # Additional web resources
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### 1. Clone/Download the Project
```bash
git clone <repository-url>
cd ET201
```

### 2. Install Dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn joblib
```

### 3. Train/Update the Model (Optional)
To retrain the Random Forest model:
```bash
python Training_Models/train_model3.py
```
This generates the `Models/RandomForest_Model.pkl` file.

### 4. Run the Backend Server
Start the Flask API:
```bash
python app.py
```
The server will start at `http://localhost:5000` and automatically load the trained model.

```
✅ Model loaded successfully from [path]/RandomForest_Model.pkl
🌊 Water Quality Prediction API
API Endpoint: http://localhost:5000/api/predict
Status Check: http://localhost:5000/api/status
```

### 5. Launch the Dashboard
Open the web dashboard in your browser:
- **Option A**: Open `Websites/dashboard.html` directly in your browser
- **Option B**: Use VS Code Live Server for better performance
- **Option C**: Run a simple HTTP server:
  ```bash
  python -m http.server 8000
  # Then visit http://localhost:8000/Websites/dashboard.html
  ```

## 🖥️ Usage Guide

### Dashboard Features

1. **User Authentication**
   - Login or register to access the dashboard
   - Store your ThingSpeak API credentials for easy IoT integration

2. **Water Analysis**
   - **Manual Input**: Enter all 9 water quality parameters
   - **Sample Data**: Click "Load Sample Data" to auto-fill with realistic test values
   - **IoT Integration**: Load real-time data directly from ThingSpeak

3. **Parameter Analysis**
   - **Safe Range Display**: See optimal and current values for each parameter
   - **Risk Assessment**: Color-coded indicators (Green=Safe, Orange=Medium, Red=High)
   - **Specific Recommendations**: Numbered action steps tailored to your water condition
   - **Health Risks**: Detailed warnings about potential health impacts

4. **Results Interpretation**
   - Overall potability verdict (Potable/Not Potable)
   - Model confidence score (0-100%)
   - Parameter-by-parameter breakdown with recommendations
   - Health risk warnings when applicable

5. **Data Management**
   - Track testing history with timestamps
   - View detailed parameter values from previous tests
   - Export complete reports as CSV files

## 📊 API Endpoints

### `POST /api/predict`
Predicts water potability based on quality parameters.

**Request Body:**
```json
{
    "ph": 7.5,
    "Hardness": 250,
    "Solids": 300,
    "Chloramines": 2.5,
    "Sulfate": 150,
    "Conductivity": 500,
    "Organic_carbon": 12,
    "Trihalomethanes": 45,
    "Turbidity": 2.5
}
```

**Response:**
```json
{
    "prediction": 1,
    "potable": true,
    "probability": 0.8742,
    "confidence": 87.42,
    "message": "Water is POTABLE ✅",
    "status": "safe",
    "parameters": { ... }
}
```

### `GET /api/status`
Checks if the API and model are running.

**Response:**
```json
{
    "status": "running",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### `GET /api/model-info`
Returns information about the trained model.

## 🎯 Water Quality Parameters

| Parameter | Unit | Safe Range | Description |
|-----------|------|-----------|-------------|
| **pH** | - | 6.5 - 8.5 | Acidity/alkalinity balance |
| **Hardness** | mg/L | 0 - 300 | Dissolved calcium & magnesium |
| **Solids (TDS)** | mg/L | 0 - 500 | Total dissolved solids |
| **Chloramines** | mg/L | 0 - 4 | Disinfection chemical level |
| **Sulfate** | mg/L | 0 - 250 | Sulfur compound concentration |
| **Conductivity** | μS/cm | 0 - 800 | Electrical conductivity |
| **Organic Carbon** | ppm | 0 - 20 | Organic matter content |
| **Trihalomethanes** | μg/L | 0 - 80 | Disinfection byproducts |
| **Turbidity** | NTU | 0 - 5 | Water clarity/cloudiness |

## 🔄 Workflow

1. **Data Input** → Enter water parameters (manual, sample, or IoT)
2. **Model Prediction** → Flask API uses Random Forest to predict potability
3. **Risk Analysis** → System evaluates each parameter against safe ranges
4. **Recommendations** → Generates specific action steps for improvements
5. **Health Warnings** → Displays potential health risks for out-of-range values
6. **Results Display** → Shows verdict, confidence, and detailed breakdown
7. **Data Storage** → Saves to history for tracking and reporting

## 🤖 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 9 water quality parameters
- **Training Data**: Comprehensive water potability dataset
- **Output**: Binary classification (Potable/Not Potable) with probability scores
- **Performance Metric**: Cross-validation ROC-AUC score for model evaluation

## 📈 Key Improvements

- ✅ Multi-parameter support (9 parameters vs. original 5)
- ✅ Detailed health risk database with specific warnings
- ✅ Actionable recommendations with exact numerical guidance
- ✅ Enhanced UI with color-coded risk levels
- ✅ Complete parameter history and CSV export
- ✅ Real-time IoT data integration
- ✅ Confidence scores for predictions

## 🤝 Contributing
Contributions are welcome! Feel free to:
- Submit bug reports or feature requests
- Fork the repository and submit pull requests
- Improve model accuracy with new training data

## 📄 License
This project is open-source and available under the MIT License.

## 👨‍💻 Author
Sashwat Puri Sachdev

## 📞 Support
For issues, questions, or suggestions, please open an issue in the repository.

