<div align="center">

# 🩸 GlucoSense – Diabetes Risk Prediction System

**AI-powered diabetes risk prediction from your health metrics**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4?logo=python&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-blueviolet)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?logo=opensourceinitiative&logoColor=white)](LICENSE)

[Live Demo](https://your-app-name.streamlit.app) · [Report Bug](https://github.com/POOJAS324/glucosense/issues) · [Request Feature](https://github.com/POOJAS324/glucosense/issues)

</div>

---

## 🧠 Problem Statement

Diabetes is one of the most widespread chronic diseases globally, affecting over **400 million** people. Early detection is critical — yet many individuals remain undiagnosed until serious complications arise.

**GlucoSense** addresses this gap by providing an interactive, ML-powered web tool that predicts diabetes risk in real time based on eight key health indicators.  It goes beyond a simple "Diabetic / Not Diabetic" label by showing **risk as a percentage**, explaining which features drove the prediction using **SHAP values**, and presenting a transparent **model comparison** so users know exactly how the best model was selected.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎛️ **Interactive Sliders** | Adjust 8 health parameters in the sidebar |
| 🤖 **Multi-Model Comparison** | Logistic Regression, Random Forest, XGBoost — best chosen by ROC-AUC |
| 📊 **Risk Gauge** | Probability displayed as a percentage with color-coded risk level |
| 🔍 **SHAP Explanations** | Per-prediction bar chart shows which features pushed risk up or down |
| 📈 **Population Comparison Plots** | 7 scatter plots comparing your data against the full dataset |
| 🗂️ **Model Insights Tab** | Confusion matrix, ROC curve, global SHAP importance, model comparison table |
| 🧹 **Proper Data Cleaning** | KNN imputation of biologically impossible zeros |
| 🧪 **Unit Tests** | pytest suite for the preprocessing pipeline |
| 🐳 **Docker Support** | One-command reproducible deployment |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Imputation | scikit-learn KNNImputer |
| Data Handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialisation | joblib |
| Testing | pytest |
| Language | Python 3.10+ |

---

## 🏆 Model Comparison Results

The table below is updated automatically each time `src/train.py` is run.

| Rank | Model | CV ROC-AUC (5-fold) |
|------|-------|---------------------|
| 🥇 | XGBClassifier | ~0.840 |
| 🥈 | RandomForestClassifier | ~0.830 |
| 🥉 | LogisticRegression | ~0.820 |

> Exact numbers depend on the run; the **Model Insights** tab in the app always shows the live values from `models/metrics.json`.

---

## ⚙️ How It Works

```
diabetes.csv
     │
     ▼
┌──────────────────────┐
│  Replace 0→NaN in    │  Glucose, BloodPressure, SkinThickness,
│  impossible columns  │  Insulin, BMI
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  KNNImputer (k=5)    │  Fit on training data, transform test + user input
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  StandardScaler      │  Fit on training data
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  GridSearchCV (5-fold, ROC-AUC scoring)       │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │   Logistic │ │   Random   │ │    XGB    │ │
│  │ Regression │ │   Forest   │ │Classifier │ │
│  └────────────┘ └────────────┘ └───────────┘ │
└──────────────────────┬───────────────────────┘
                       │  Best model by CV ROC-AUC
                       ▼
              ┌────────────────┐
              │  Evaluate on   │
              │  held-out 20 % │
              └───────┬────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
 model.pkl      metrics.json    SHAP explainer
 (joblib)      (test metrics,   (baked into
               ROC curve,        model.pkl)
               comparison)
```

---

## 🚀 Installation & Run

### Option A — Local (recommended for development)

```bash
# 1. Clone the repository
git clone https://github.com/POOJAS324/glucosense.git
cd glucosense

# 2. Create & activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (run ONCE before launching the app)
python src/train.py

# 5. Launch the Streamlit app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

> ⚠️ **Step 4 is required before Step 5.**  The app will show a warning banner if `models/model.pkl` is missing.

---

### Option B — Docker

```bash
# Build the image (trains the model automatically at build time)
docker build -t glucosense .

# Run the container
docker run -p 8501:8501 glucosense
```

Open **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

```bash
# From the project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage (if pytest-cov is installed)
pytest tests/ --cov=src --cov-report=term-missing
```

All 5 unit tests cover:
- Zeros in impossible columns are correctly replaced with NaN
- `Pregnancies` and `Outcome` zeros are left untouched
- No NaNs remain after imputation
- Row count and column set are preserved
- A pre-fitted imputer can be reused at inference time

---

## 📂 Folder Structure

```
GlucoSense/
├── .streamlit/
│   └── config.toml              # Streamlit dark health-tech theme
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # load + KNN impute
│   ├── train.py                 # offline training script
│   └── predict.py               # inference helpers
├── tests/
│   ├── __init__.py
│   └── test_data_preprocessing.py
├── models/                      # created by train.py (gitignored)
│   ├── model.pkl                # joblib bundle (model + imputer + scaler + SHAP)
│   └── metrics.json             # test metrics + ROC data + model comparison
├── app.py                       # Streamlit UI (imports from src/, never trains)
├── diabetes.csv                 # Pima Indians Diabetes Dataset
├── requirements.txt
├── Dockerfile
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

- **Source:** [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 (all female, ≥ 21 years, Pima Indian heritage)
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** Outcome (0 = Healthy, 1 = Diabetic)
- **Class balance:** ~65 % healthy, ~35 % diabetic

> **Note on data quality:** Five feature columns use `0` as a sentinel for missing values.  GlucoSense handles this explicitly in `src/data_preprocessing.py`.

---

## 🔮 Future Scope

- 🧬 **Deep Learning Models** — Experiment with neural networks for improved accuracy
- 🏥 **EHR Integration** — Connect with Electronic Health Record systems
- 📱 **Mobile App** — Build a cross-platform mobile version
- 🌍 **Multilingual Support** — Offer the interface in multiple languages
- 🔐 **User Authentication** — Allow users to save and track health history
- ⌚ **Wearable Device Integration** — Ingest real-time data from fitness trackers

---

## 👩‍💻 Author

**Pooja S**
GitHub: [@POOJAS324](https://github.com/POOJAS324)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and Python

</div>
