# 🩸 GlucoSense – Diabetes Risk Prediction System

An AI-powered web application that predicts diabetes risk using machine learning based on key health parameters. The application provides diabetes risk probability, explains predictions using SHAP, and compares multiple ML models to improve transparency.

[Live Demo](https://glucosense-app.streamlit.app) · [Report Bug](https://github.com/POOJAS324/GlucoSense/issues) · [Request Feature](https://github.com/POOJAS324/GlucoSense/issues)

---

## ✨ Features

- 🤖 Predicts diabetes risk using Machine Learning
- 📊 Displays risk probability with interactive visualization
- 🔍 Explains predictions using SHAP feature importance
- 📈 Compares Logistic Regression, Random Forest, and XGBoost models
- 🧹 Handles missing values using KNN Imputation
- 🧪 Includes unit tests for preprocessing
- 🐳 Docker support for easy deployment

---

## 🛠️ Tech Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Testing | Pytest |
| Language | Python 3.10+ |

---

## 🚀 Installation & Run

```bash
# Clone the repository
git clone https://github.com/POOJAS324/GlucoSense.git
cd GlucoSense

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Run the application
streamlit run app.py
```

The application will open at **http://localhost:8501**

---

## 📂 Project Structure

```text
GlucoSense/
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── tests/
├── models/
├── app.py
├── diabetes.csv
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

---

## 📊 Dataset

- **Dataset:** Pima Indians Diabetes Dataset
- **Records:** 768
- **Features:** 8 health-related attributes
- **Target:** Diabetes Prediction (0 = Healthy, 1 = Diabetic)

---

## 🔮 Future Scope

- Deep Learning models
- User Authentication
- Mobile Application
- EHR Integration
- Wearable Device Integration
- Multilingual Support

---

## 👩‍💻 Author

**Pooja S**

GitHub: https://github.com/POOJAS324

---

## 📄 License

This project is licensed under the **MIT License**.

---

Made with ❤️ using Python.
