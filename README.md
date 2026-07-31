# HeartDiseaseDeployment — End-to-End ML Model Deployment (GitHub + Render)

## Live Deployment Link

https://heart-disease-prediction-dq11.onrender.com/ui

Name: Divyansh Kumar

Registration No.: 23BAI10514

Application No.: IN26011845

Batch No.: 1 A

Email ID: divyansh.23bai10514@vitbhopal.ac.in

## 🎯 Objective

Build a machine learning model that predicts whether a patient is at risk of heart disease based on clinical parameters, expose it through a Flask REST API, and deploy it as a live web service on Render.

## 📊 Dataset Link

**Heart Disease Prediction Dataset (Kaggle):**
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

> ⚠️ The full dataset is not required in the repo beyond `heart.csv` (see [Repository Structure](#-repository-structure)). Download it via `kagglehub` (used in the notebook) or the Kaggle CLI.

## 🧰 Libraries Used

- `pandas`, `numpy` — data loading & preprocessing
- `scikit-learn` — Random Forest classifier, train/test split, evaluation metrics
- `joblib` — model serialization
- `flask` — REST API
- `gunicorn` — production WSGI server (used by Render)
- `matplotlib`, `seaborn` — visualization (notebook only)
- `kagglehub` — dataset download (notebook only)

## 🧪 Methodology

1. **Data Understanding & Preprocessing** — Loaded `heart.csv` with Pandas, inspected the first five records, identified the 13 numerical clinical features and the binary `target` variable, confirmed there are no missing values, and split the data 80/20 into train/test sets (stratified on the target).
2. **Model Development** — Trained a `RandomForestClassifier` (200 trees, max depth 8) on the training set, evaluated it on the held-out test set using accuracy plus a full classification report and confusion matrix, and saved the trained model with Joblib (`model.pkl`), along with the exact feature column order (`feature_order.json`) so the API can build correctly-ordered input rows.
3. **API Development** — Built a Flask REST API (`app.py`) that loads `model.pkl` at startup, accepts patient details as a JSON POST body at `/predict`, validates the input, and returns a JSON prediction.
4. **GitHub & Cloud Deployment** — Pushed the repository to GitHub and deployed the Flask app on Render using Gunicorn (see [Deployment](#-deployment-render) below).
5. **Conclusion** — Summarized model performance, deployment challenges, and the role of MLOps.

## 🏗️ Model

- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `max_depth=8`, `random_state=42`)
- **Features (13):** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target:** `target` (1 = heart disease present, 0 = no heart disease)
- **Evaluation Metric:** Accuracy Score (plus precision/recall/F1 in the notebook for completeness)

## 📈 Results

_Fill in after running `train_model.py` / the notebook on the full dataset:_

| Metric | Value |
|--------|-------|
| Test Accuracy | 0.9902 |

See `Assignment-10.ipynb` for the confusion matrix, full classification report, and feature-importance plot.

## 🔌 API Usage

**Base URL (local):** `http://127.0.0.1:5000`
**Base URL (deployed):** `https://heart-disease-prediction-dq11.onrender.com`

### `GET /`
Health check — returns a message and the list of expected input fields.

### `GET /ui`
Optional browser-based form (`templates/index.html`) — fill in patient details and submit to see a prediction rendered on the page. It calls `/predict` behind the scenes via `fetch()`.

### `POST /predict`
Accepts patient details as JSON and returns a prediction.

**Request:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**
```json
{
  "prediction": "Heart Disease Detected",
  "prediction_label": 1,
  "probability": {
    "no_disease": 0.18,
    "disease": 0.82
  }
}
```

**cURL example:**
```bash
curl -X POST https://https://heart-disease-prediction-dq11.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## 🚀 Setup / How to Run Locally

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd HeartDiseaseDeployment
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Get `heart.csv` (via `kagglehub` — see `Assignment-10.ipynb` — or the Kaggle CLI) and place it in the project root.
4. Train the model (produces `model.pkl` and `feature_order.json`):
   ```bash
   python train_model.py
   ```
5. Run the API:
   ```bash
   python app.py
   ```
6. Test it with the cURL command above (pointed at `http://127.0.0.1:5000/predict`).

## ☁️ Deployment (Render)

1. **Push to GitHub** — make sure `app.py`, `model.pkl`, `requirements.txt`, `Procfile`, and `train_model.py` are all committed and the repo is public:
   ```bash
   git init
   git add .
   git commit -m "Heart disease prediction API"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. **Create a new Web Service on Render** — two options:

   **Option A — Dashboard (manual):**
   - Go to [render.com](https://render.com) → **New** → **Web Service**.
   - Connect your GitHub repository.
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set in the `Procfile`, but Render also lets you set it explicitly in the dashboard)
   - **Instance Type:** Free tier is fine for evaluation purposes.

   **Option B — Blueprint (one-click via `render.yaml`):**
   - Go to [render.com](https://render.com) → **New** → **Blueprint**.
   - Connect your repository — Render reads `render.yaml` and provisions the service automatically (build/start commands, Python version) without manual field entry.
3. **Deploy** — Render will build and deploy automatically. Once live, you'll get a public URL like `https://heartdiseasedeployment.onrender.com`.
4. **Verify:**
   ```bash
   curl https://heartdiseasedeployment.onrender.com/
   curl -X POST https://heartdiseasedeployment.onrender.com/predict -H "Content-Type: application/json" -d '{...}'
   ```
5. **Keep it active** — Render's free tier spins down idle services and takes ~30–60s to "wake up" on the next request; if evaluation timing matters, either upgrade the plan or ping the endpoint shortly before evaluation to warm it up.
6. Paste your live URL here: **`https://heart-disease-prediction-dq11.onrender.com`**

## ✅ Conclusion

This project trained a Random Forest classifier to predict heart disease risk from 13 clinical parameters, achieving a test accuracy of approximately 99.02%. The model was serialized with Joblib and wrapped in a Flask REST API that accepts patient details as JSON and returns a prediction. The main deployment challenges involved keeping the incoming JSON feature order aligned with the order the model was trained on, pinning compatible library versions between the training and serving environments to avoid unpickling errors, and configuring Render's build/start commands and free-tier cold-start behavior for a reliably reachable endpoint. This exercise highlighted why MLOps practices matter: version-controlled code and models, reproducible environments via `requirements.txt`, and a clean separation between training and serving logic are what make a model dependable once it leaves the notebook and reaches real users.

## 📁 Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                    # Flask REST API
├── model.pkl                 # Trained Random Forest model (Joblib)
├── feature_order.json        # Exact feature column order expected by the model
├── requirements.txt          # Python dependencies
├── Procfile                  # Render/Gunicorn start command
├── render.yaml                # Optional: one-click Render Blueprint config
├── train_model.py            # Standalone training script
├── Assignment-10.ipynb       # Full task-by-task notebook (Tasks 1, 2, 5)
├── heart.csv                 # Dataset (place after downloading from Kaggle)
├── README.md                 # This file
├── templates/
│   └── index.html             # Optional browser-based prediction form (served at /ui)
└── static/                   # (Optional, unused by default)
```

## 👤 Author

**Name:** Divyansh Kumar

**Registration Number:** 23BAI10514

**Render Deployment URL:** `https://heart-disease-prediction-dq11.onrender.com`
