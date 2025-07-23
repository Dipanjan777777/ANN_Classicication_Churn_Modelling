# Customer Churn Prediction

## Problem Statement
The goal of this project is to predict whether a bank customer is likely to churn (leave the bank) based on their personal and financial information. Churn prediction helps banks identify at-risk customers and take proactive measures to retain them.

## Approach
1. **Data Preprocessing**:  
   - Remove unnecessary columns.
   - Encode categorical variables (Gender with LabelEncoder, Geography with OneHotEncoder).
   - Scale numerical features using StandardScaler.

2. **Model Building**:  
   - Train an Artificial Neural Network (ANN) using TensorFlow/Keras.
   - Use early stopping and TensorBoard for monitoring training.

3. **Deployment**:  
   - Build a Streamlit web app for user-friendly predictions.
   - Save preprocessing objects and the trained model for inference.

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <repo-url>
cd Churn_Modelling
```

### 2. Install uv (Python package manager)
If you don't have `uv` installed, follow instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

### 3. Create and Activate a Virtual Environment
```sh
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Dependencies
```sh
uv pip install -r requirements.txt
```

### 5. Run the Streamlit App
```sh
streamlit run app.py
```

### 6. (Optional) Run Jupyter Notebooks
```sh
uv pip install notebook
jupyter notebook
```

## Files
- `app.py`: Streamlit web application for predictions.
- `experiments.ipynb`: Data preprocessing and model training notebook.
- `prediction.ipynb`: Model inference notebook.
- `requirements.txt`: List of required Python packages.

## Notes
- Ensure you have the dataset `Churn_Modelling.csv` in the project directory.
- Preprocessing objects and model files (`*.pkl`, `churn_model.h5`) are generated during notebook execution.
