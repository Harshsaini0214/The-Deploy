
# Sales Prediction Using Machine Learning

A machine learning web application that predicts sales based on advertising expenditure across TV, radio, and newspaper channels. Built with Python, FastAPI, and scikit-learn, the application provides an interactive browser dashboard, compares three regression models, and displays model evaluation metrics.

## Overview

This project demonstrates an end-to-end machine learning workflow, from loading and preprocessing a dataset to training regression models, evaluating their performance, and serving predictions through a web API.

The original Streamlit-based approach has been adapted into a FastAPI application with a browser-based frontend, making it suitable for deployment on Vercel.

### Key Features

- Predict sales using TV, radio, and newspaper advertising budgets.
- Compare Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
- Automatically select the best-performing model based on test-set R².
- View model evaluation metrics and dataset statistics.
- Generate predictions through a REST API.
- Store recent prediction history in browser localStorage.
- Access interactive API documentation through Swagger UI.
- Deploy the application on Vercel.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Data Preprocessing | StandardScaler |
| Regression Models | Linear Regression, Decision Tree, Random Forest |
| Model Evaluation | MAE, MSE, RMSE, R² Score |
| API Documentation | FastAPI / Swagger UI |
| Deployment | Vercel |
| Dataset | CSV |

## Dataset

The project uses `dataset/advertising.csv`, which contains advertising expenditure and corresponding sales values.

The dataset contains four columns:

| Column | Description |
|---|---|
| `TV` | Advertising expenditure on TV |
| `Radio` | Advertising expenditure on radio |
| `Newspaper` | Advertising expenditure on newspapers |
| `Sales` | Recorded sales value and prediction target |

The three advertising channels are the input features, while `Sales` is the target variable.

### Data Preprocessing

The application performs the following operations:

1. Loads the CSV dataset using Pandas.
2. Validates the required columns.
3. Converts feature and target values to numeric types.
4. Removes invalid, missing, infinite, and duplicate records.
5. Prepares the data for model training.
6. Checks that sufficient valid records remain.

## Machine Learning Models

The application compares three regression algorithms.

### 1. Linear Regression

Linear Regression estimates the relationship between advertising expenditure and sales using a linear equation.

### 2. Decision Tree Regressor

A Decision Tree Regressor learns decision rules from the training data to predict sales.

### 3. Random Forest Regressor

Random Forest combines multiple decision trees to produce predictions and model nonlinear relationships.

### Model Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error measures the average absolute prediction error. |
| MSE | Mean Squared Error measures the average squared prediction error. |
| RMSE | Root Mean Squared Error is the square root of MSE. |
| R² Score | Measures how well the model explains variation in the target values. |

The application evaluates all three models on a held-out test set and selects the model with the highest R² score as the default model.

## Project Structure

```text
Sales-Prediction-Vercel-Patch/
├── app.py
├── index.html
├── requirements.txt
├── vercel.json
├── .python-version
└── dataset/
    └── advertising.csv
```

### File Description

- `app.py` — FastAPI backend, dataset processing, model training, evaluation, and prediction endpoints.
- `index.html` — Browser-based user interface.
- `requirements.txt` — Python dependencies.
- `vercel.json` — Vercel deployment configuration.
- `.python-version` — Specifies the Python runtime.
- `dataset/advertising.csv` — Advertising and sales dataset.

## Getting Started Locally

### Prerequisites

- Python 3.12 recommended
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Harshsaini0214/First-Deployment
```

Navigate to the project directory:

```bash
cd First Deployment/Sales-Prediction-Vercel-Patch
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install uvicorn
```

### 4. Run the Application

```bash
python -m uvicorn app:app --reload
```

### 5. Open the Application

Once the server starts, access the following URLs:

| Resource | URL |
|---|---|
| Web Application | http://127.0.0.1:8000/ |
| Health Check | http://127.0.0.1:8000/api/health |
| Dataset and Model Summary | http://127.0.0.1:8000/api/summary |
| Swagger API Documentation | http://127.0.0.1:8000/docs |
| ReDoc API Documentation | http://127.0.0.1:8000/redoc |

## API Reference

The FastAPI backend exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web interface. |
| GET | `/api/health` | Returns application health information. |
| GET | `/api/summary` | Returns dataset statistics and model evaluation results. |
| POST | `/api/predict` | Predicts sales from advertising expenditure. |
| GET | `/docs` | Opens Swagger UI documentation. |
| GET | `/redoc` | Opens ReDoc documentation. |

### Prediction API Example

Send a POST request to `/api/predict`.

Example JSON request:

```json
{
  "tv": 150,
  "radio": 25,
  "newspaper": 30,
  "model_name": "Linear Regression"
}
```

The `model_name` field is optional. If omitted or set to `null`, the application uses the best-performing model.

Supported model names:

- `Linear Regression`
- `Decision Tree Regressor`
- `Random Forest Regressor`

Example using cURL:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tv":150,"radio":25,"newspaper":30}'
```

A successful response includes the predicted sales value, selected model, evaluation metrics, and input values.

Example response format:

```json
{
  "prediction": 16.5,
  "model_name": "Linear Regression",
  "metrics": {
    "MAE": 1.0,
    "MSE": 1.5,
    "RMSE": 1.2247,
    "R2 Score": 0.9
  },
  "inputs": {
    "TV": 150,
    "Radio": 25,
    "Newspaper": 30
  }
}
```

**Note:** The numerical values above are illustrative examples of the response format, not actual measured model results.

## Important Notes and Limitations

- Models are trained from the dataset and cached within the current server process.
- Vercel may start new server processes, which can require the models to be trained again.
- The quality of predictions depends on the supplied dataset and the distribution of the training data.
- The 80:20 train-test split provides a basic evaluation but does not guarantee future performance.
- Advertising budgets are restricted to the range accepted by the API; valid input does not guarantee accurate predictions.
- The project is intended for educational and demonstration purposes.

## Future Improvements

- Add cross-validation and more comprehensive model evaluation.
- Visualize actual versus predicted sales and residual errors.
- Introduce feature importance analysis where appropriate.
- Add database-backed prediction history.
- Improve automated testing and deployment validation.
- Evaluate the models using additional datasets.

## Learning Outcomes

This project demonstrates practical experience with:

- Data cleaning and preprocessing.
- Regression modelling with scikit-learn.
- Model evaluation and comparison.
- REST API development with FastAPI.
- Frontend and backend integration.
- API input validation.
- Deploying a Python machine learning application on Vercel.

## Repository

**GitHub Repository:**  
https://github.com/Harshsaini0214/First-Deployment

**Vercel Link:**
https://sales-predict.vercel.app/

## Author

**Harsh Saini**

*Bachelor of Computer Applications (Artificial Intelligence & Machine Learning)*   
GitHub: [@Harshsaini0214](https://github.com/Harshsaini0214)

---

*Developed for learning, experimentation, and demonstrating machine learning application deployment.*
