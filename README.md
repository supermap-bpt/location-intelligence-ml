# 🧠 Suitability Prediction API (FastAPI)

This project provides a REST API to predict the **suitability category** of a geographic location using a trained classification model.

---

## 🚀 Features

- Classifies a polygon area as:
  - ✅ `recommended`
  - 😐 `neutral`
  - ❌ `not recommended`
- Based on spatial feature scores and user-defined weights
- Returns model confidence and metadata

---

## 📦 Setup Instructions (using `venv`)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/suitability-api.git
cd suitability-api
```

### 2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate    # On macOS/Linux
# OR
venv\Scripts\activate       # On Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```
---
## Directory Structure
```
project/
│
├── model/
│   ├── suitability_classifier.pkl
│
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```
---
## 🚦 Running the API Server
```
uvicorn main:app --reload
```

This will start the server at:
```
http://127.0.0.1:8000
```

Interactive Swagger docs:
```
http://127.0.0.1:8000/docs
```

---
## Evaluation Results

```
--- Classification Report ---

                 precision    recall  f1-score   support

Not Recommended       1.00      0.89      0.94         9
        Neutral       0.95      0.86      0.90        22
    Recommended       0.88      1.00      0.94        22

       accuracy                           0.92        53
      macro avg       0.94      0.92      0.93        53
   weighted avg       0.93      0.92      0.92        53


```

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 0.92  |
| Macro F1-score  | 0.93  |
| Precision (avg) | 0.94  |
| Recall (avg)    | 0.92  |
