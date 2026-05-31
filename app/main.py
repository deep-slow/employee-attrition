from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os
import io
from difflib import SequenceMatcher

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict employee attrition using a stacking ensemble model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load stacking model ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stacking_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    stacking = pickle.load(f)

xgb_model   = stacking['xgb']
rf_model    = stacking['rf']
lr_model    = stacking['lr']
meta_model  = stacking['meta_model']
best_threshold = stacking['best_threshold']

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Human Resources', 'Department_Research & Development', 'Department_Sales',
    'EducationField_Human Resources', 'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
    'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'CompensationRatio', 'TenurePerJob', 'YearsWithoutChange'
]

# 30 raw input fields HR provides
REQUIRED_FIELDS = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'
]

# Categorical fields — fail if missing (can't guess these)
CATEGORICAL_FIELDS = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime']

# Numerical defaults (median from IBM HR dataset)
NUMERICAL_DEFAULTS = {
    'Age': 36, 'DailyRate': 802, 'DistanceFromHome': 7, 'Education': 3,
    'EnvironmentSatisfaction': 3, 'HourlyRate': 66, 'JobInvolvement': 3,
    'JobLevel': 2, 'JobSatisfaction': 3, 'MonthlyIncome': 4919,
    'MonthlyRate': 14235, 'NumCompaniesWorked': 2, 'PercentSalaryHike': 14,
    'PerformanceRating': 3, 'RelationshipSatisfaction': 3, 'StockOptionLevel': 1,
    'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
    'YearsAtCompany': 5, 'YearsInCurrentRole': 3, 'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 3
}

# ── Helper: fuzzy column matching ────────────────────────────────────────────
def fuzzy_match(col: str, candidates: list, threshold: float = 0.6) -> str | None:
    col_lower = col.lower().replace('_', '').replace(' ', '')
    best_score = 0
    best_match = None
    for candidate in candidates:
        cand_lower = candidate.lower().replace('_', '').replace(' ', '')
        score = SequenceMatcher(None, col_lower, cand_lower).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score >= threshold else None

def map_columns(df_cols: list) -> dict:
    mapping = {}
    for col in df_cols:
        match = fuzzy_match(col, REQUIRED_FIELDS)
        if match:
            mapping[col] = match
    return mapping

# ── Helper: build feature array from raw dict ─────────────────────────────────
def build_features(data: dict) -> np.ndarray:
    features = {}

    # Numerical
    for field in NUMERICAL_DEFAULTS:
        features[field] = float(data.get(field, NUMERICAL_DEFAULTS[field]))

    # Binary encode
    features['Gender']  = 1 if str(data.get('Gender', '')).strip().lower() == 'male' else 0
    features['OverTime'] = 1 if str(data.get('OverTime', '')).strip().lower() == 'yes' else 0

    # One-hot — BusinessTravel
    bt = str(data.get('BusinessTravel', '')).strip()
    features['BusinessTravel_Non-Travel']        = 1 if bt == 'Non-Travel' else 0
    features['BusinessTravel_Travel_Frequently'] = 1 if bt == 'Travel_Frequently' else 0
    features['BusinessTravel_Travel_Rarely']     = 1 if bt == 'Travel_Rarely' else 0

    # One-hot — Department
    dept = str(data.get('Department', '')).strip()
    features['Department_Human Resources']       = 1 if dept == 'Human Resources' else 0
    features['Department_Research & Development']= 1 if dept == 'Research & Development' else 0
    features['Department_Sales']                 = 1 if dept == 'Sales' else 0

    # One-hot — EducationField
    for field in ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']:
        features[f'EducationField_{field}'] = 1 if data.get('EducationField', '') == field else 0

    # One-hot — JobRole
    for role in ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                 'Manager', 'Manufacturing Director', 'Research Director',
                 'Research Scientist', 'Sales Executive', 'Sales Representative']:
        features[f'JobRole_{role}'] = 1 if data.get('JobRole', '') == role else 0

    # One-hot — MaritalStatus
    for status in ['Divorced', 'Married', 'Single']:
        features[f'MaritalStatus_{status}'] = 1 if data.get('MaritalStatus', '') == status else 0

    # Engineered features
    monthly_income   = float(data.get('MonthlyIncome', NUMERICAL_DEFAULTS['MonthlyIncome']))
    job_level        = float(data.get('JobLevel', NUMERICAL_DEFAULTS['JobLevel']))
    years_at_company = float(data.get('YearsAtCompany', NUMERICAL_DEFAULTS['YearsAtCompany']))
    num_companies    = float(data.get('NumCompaniesWorked', NUMERICAL_DEFAULTS['NumCompaniesWorked']))
    years_in_role    = float(data.get('YearsInCurrentRole', NUMERICAL_DEFAULTS['YearsInCurrentRole']))
    years_promoted   = float(data.get('YearsSinceLastPromotion', NUMERICAL_DEFAULTS['YearsSinceLastPromotion']))

    features['CompensationRatio'] = monthly_income / (job_level + 1)
    features['TenurePerJob']      = years_at_company / (num_companies + 1)
    features['YearsWithoutChange']= years_in_role + years_promoted

    return np.array([[features[f] for f in FEATURE_ORDER]])

# ── Helper: run stacking prediction ──────────────────────────────────────────
def run_prediction(input_array: np.ndarray) -> dict:
    p_xgb = xgb_model.predict_proba(input_array)[:, 1]
    p_rf  = rf_model.predict_proba(input_array)[:, 1]
    p_lr  = lr_model.predict_proba(input_array)[:, 1]

    meta_features = np.column_stack((p_xgb, p_rf, p_lr))
    probability   = float(meta_model.predict_proba(meta_features)[0][1])
    prediction    = int(probability >= best_threshold)

    risk_tier = (
        "High"   if probability >= 0.65 else
        "Medium" if probability >= 0.35 else
        "Low"
    )

    return {
        "prediction":   "Will Leave" if prediction == 1 else "Will Stay",
        "probability":  round(probability * 100, 2),
        "risk_tier":    risk_tier,
        "threshold":    round(best_threshold, 2)
    }

# ── Pydantic model for single predict ────────────────────────────────────────
class EmployeeInput(BaseModel):
    Age: int
    DailyRate: int
    DistanceFromHome: int
    Education: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    BusinessTravel: str
    Department: str
    EducationField: str
    JobRole: str
    MaritalStatus: str

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API", "version": "1.0.0", "docs": "/docs"}

@app.post("/predict", summary="Predict attrition for a single employee")
def predict(emp: EmployeeInput):
    """
    Accepts 30 employee features, returns:
    - prediction: Will Leave / Will Stay
    - probability: % chance of leaving
    - risk_tier: Low / Medium / High
    - threshold: decision threshold used
    """
    try:
        input_array = build_features(emp.dict())
        return run_prediction(input_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-bulk", summary="Predict attrition for multiple employees via CSV upload")
async def predict_bulk(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Fuzzy match columns
        col_mapping = map_columns(df.columns.tolist())
        df = df.rename(columns=col_mapping)

        # Check missing categoricals — these we cannot guess
        missing_categoricals = [f for f in CATEGORICAL_FIELDS if f not in df.columns]
        if missing_categoricals:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required categorical columns that cannot be inferred",
                    "missing": missing_categoricals,
                    "hint": "Download /sample-csv to see required column names"
                }
            )

        # Check invalid categorical values
        VALID_VALUES = {
            "Department": ["Sales", "Research & Development", "Human Resources"],
            "Gender": ["Male", "Female"],
            "OverTime": ["Yes", "No"],
            "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            "MaritalStatus": ["Single", "Married", "Divorced"],
            "JobRole": [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ],
            "EducationField": [
                "Life Sciences", "Medical", "Marketing",
                "Technical Degree", "Human Resources", "Other"
            ]
        }

        invalid_values = {}
        for field, valid in VALID_VALUES.items():
            if field in df.columns:
                bad = df[~df[field].isin(valid)][field].unique().tolist()
                if bad:
                    invalid_values[field] = bad

        if invalid_values:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid categorical values found",
                    "invalid": invalid_values,
                    "hint": "Check valid values for each field in /docs"
                }
            )

        # Check missing numericals — warn but fill with defaults
        missing_numericals = [f for f in NUMERICAL_DEFAULTS if f not in df.columns]

        predictions = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            input_array = build_features(row_dict)
            result = run_prediction(input_array)
            predictions.append(result)

        # Add results to dataframe
        df['Prediction']  = [p['prediction']  for p in predictions]
        df['Probability'] = [p['probability'] for p in predictions]
        df['RiskTier']    = [p['risk_tier']   for p in predictions]

        # Return as downloadable CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        response_headers = {
            "Content-Disposition": "attachment; filename=attrition_predictions.csv"
        }

        meta = {}
        if missing_numericals:
            meta["warning"] = f"Missing numerical columns filled with defaults: {missing_numericals}"

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers=response_headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-csv", summary="Download a sample CSV template with example employee data")
def sample_csv():
    """
    Returns a CSV template with correct column headers and 3 example rows.
    Fill your employee data in this format and upload to /predict-bulk.
    """
    sample_data = {
        'Age':                    [35, 28, 52],
        'DailyRate':              [1102, 279, 1373],
        'DistanceFromHome':       [1, 8, 2],
        'Education':              [2, 3, 4],
        'EnvironmentSatisfaction':[2, 3, 1],
        'Gender':                 ['Male', 'Female', 'Male'],
        'HourlyRate':             [94, 61, 48],
        'JobInvolvement':         [3, 2, 3],
        'JobLevel':               [2, 1, 3],
        'JobSatisfaction':        [4, 2, 1],
        'MonthlyIncome':          [5993, 2090, 9526],
        'MonthlyRate':            [19479, 9813, 10522],
        'NumCompaniesWorked':     [8, 1, 0],
        'OverTime':               ['Yes', 'No', 'No'],
        'PercentSalaryHike':      [11, 23, 15],
        'PerformanceRating':      [3, 4, 3],
        'RelationshipSatisfaction':[1, 4, 2],
        'StockOptionLevel':       [0, 1, 0],
        'TotalWorkingYears':      [8, 10, 6],
        'TrainingTimesLastYear':  [0, 3, 3],
        'WorkLifeBalance':        [1, 3, 3],
        'YearsAtCompany':         [6, 10, 0],
        'YearsInCurrentRole':     [4, 7, 0],
        'YearsSinceLastPromotion':[0, 1, 0],
        'YearsWithCurrManager':   [5, 7, 0],
        'BusinessTravel':         ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
        'Department':             ['Sales', 'Research & Development', 'Human Resources'],
        'EducationField':         ['Life Sciences', 'Medical', 'Marketing'],
        'JobRole':                ['Sales Executive', 'Research Scientist', 'Laboratory Technician'],
        'MaritalStatus':          ['Single', 'Married', 'Divorced'],
    }

    df = pd.DataFrame(sample_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_employees.csv"}
    )


@app.get("/health", summary="Check if API is running")
def health():
    return {
        "status": "ok",
        "model": "Stacking Ensemble (XGB + RF + LR)",
        "threshold": round(best_threshold, 2),
        "features_expected": 52,
        "input_fields": 30
    }
