# 🌊 AquaSense – AI & IoT-Based Water Quality Management System

An end-to-end **AI and IoT-driven water quality analysis system** designed to evaluate water potability, assess risk levels, map potential health impacts, and recommend corrective actions using real-time and manual water parameter inputs.

This project combines **machine learning model comparison**, **handling of imbalanced data**, **IoT integration via ThingSpeak**, and an **interactive dashboard** to deliver explainable and actionable water quality intelligence.

---

## 📌 Project Motivation

Access to safe drinking water is a critical public health requirement. Manual water testing methods are often slow, non-predictive, and reactive. AquaSense addresses this gap by providing:

* Real-time water quality monitoring
* AI-based potability prediction
* Risk categorisation with health awareness
* Actionable recommendations for water treatment

---

## 🎯 Key Objectives

* Predict **Drinkable / Not Drinkable** water using ML
* Perform **comparative analysis** of multiple ML models
* Handle **class imbalance** using SMOTE
* Integrate **real-time IoT data** from ThingSpeak
* Provide **risk-based disease mapping** (rule-based)
* Recommend **parameter-wise corrective actions**
* Visualise insights through a dashboard

---

## 🧠 Machine Learning Models Used

A comparative study was performed using the following models:

* **K-Nearest Neighbours (KNN)** – distance-based baseline model
* **Decision Tree** – interpretable rule-based classifier
* **Random Forest** – ensemble model with reduced overfitting
* **Gradient Boosting** – a sequential ensemble learning model

📌 **Random Forest** achieved the best overall performance (especially F1-score) and was selected for deployment.

---

## ⚙️ Data Processing Pipeline

1. Dataset loading (`water_potability.csv`)
2. Missing value handling
3. Feature scaling
4. Train–test split
5. SMOTE is applied **only to the training data**
6. Model training & evaluation
7. Model serialisation (`.pkl`)

⚠️ The test dataset is kept completely untouched to prevent data leakage.

---

## 🌐 IoT Integration

* Real-time water quality values fetched using **ThingSpeak API**
* Uses:

  * Channel ID
  * Read API Key
* Supports:

  * Live sensor-based inputs
  * Manual user input via the dashboard

---

## 📊 Dashboard Capabilities

* Real-time water quality prediction
* Parameter-wise status classification:

  * Safe
  * Near Optimal
  * Medium Risk
  * High Risk
* Disease association based on unsafe parameters
* Clear maintenance and treatment recommendations

🧠 Disease mapping is **rule-based**, derived from water quality standards and domain knowledge — not ML prediction.

---

## 📂 Project Structure

```
ET201/
│
├── Data/
│   └── water_potability.csv          # Dataset
│
├── Models/
│   └── RandomForest_M1.pkl           # Best performing trained model
│
├── Notebooks/
│   └── Water_System.ipynb            # EDA + model comparison
│
├── Training_Models/
│   ├── train_model.py                # Model training script 1
│   ├── train_model2.py               # Model training script 2
│   └── train_model3.py               # Model training script 3
│
├── app.py                            # Backend logic / model inference
├── dashboard.html                   # Dashboard UI
├── README.md                        # Project documentation
```

---

## 🛠️ Tech Stack

* **Programming:** Python
* **ML Libraries:** scikit-learn, imbalanced-learn (SMOTE)
* **Data Handling:** Pandas, NumPy
* **Visualisation:** Matplotlib, Seaborn
* **IoT Platform:** ThingSpeak
* **Frontend:** HTML / Dashboard UI

---

## 🚀 How to Run the Project

1. Clone the repository
2. Install dependencies
3. Run model training scripts (optional)
4. Start the application
5. Open `dashboard.html` in browser

---

## 🔮 Future Enhancements

* Time-series forecasting of water quality trends
* Multi-class risk prediction
* Cloud deployment
* Automated alert system
* Mobile application integration

---

## 📌 Conclusion

AquaSense demonstrates the effective integration of **machine learning, IoT, and domain knowledge** to build a reliable, explainable, and real-time water quality management system. The project emphasises not only prediction accuracy but also **interpretability and actionable intelligence**, making it suitable for real-world deployment.

---

## 👤 Author

**Sashwat Puri Sachdev**
B.Tech – Computer Science
NIIT University
