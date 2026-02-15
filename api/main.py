from typing import Any, Dict, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Credit Risk API")


class RiskInput(BaseModel):
    person_age: float = Field(...)
    person_income: float = Field(...)
    person_home_ownership: str = Field(...)
    person_emp_length: float = Field(...)
    loan_intent: str = Field(...)
    loan_grade: str = Field(...)
    loan_amnt: float = Field(...)
    loan_int_rate: float = Field(...)
    loan_percent_income: float = Field(...)
    cb_person_default_on_file: str = Field(...)
    cb_person_cred_hist_length: float = Field(...)


# Global model container
MODEL: Any = None
EXPLAINER: Any = None
FEATURE_NAMES: list[str] | None = None
PREPROCESSOR: Any = None


def get_model_path() -> Path:
    base = Path(__file__).resolve().parent.parent
    primary = base / "model" / "risk_model.pkl"
    fallback = base / "model" / "rf_pipeline.joblib"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No trained model found at model/risk_model.pkl or model/rf_pipeline.joblib")


def load_trained_model() -> Any:
    path = get_model_path()
    return joblib.load(path)


def predict_record(model: Any, record: Dict[str, Any]) -> Tuple[int, float]:
    """Return predicted class and confidence for given record."""
    df = pd.DataFrame([record])
    preds = model.predict(df)
    pred_class = int(preds[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        confidence = float(max(proba))
    return pred_class, confidence


def build_shap_explainer(model: Any) -> None:
    """Initialize a SHAP TreeExplainer using a small background dataset.

    If SHAP isn't available or explainer can't be constructed, EXPLAINER remains None.
    """
    global EXPLAINER, FEATURE_NAMES, PREPROCESSOR
    try:
        import shap
        import numpy as np
    except Exception:
        # shap not installed
        EXPLAINER = None
        return

    # If model is a pipeline with preprocessor and classifier steps, extract them
    pre = None
    clf = None
    if hasattr(model, "named_steps"):
        pre = model.named_steps.get("preprocessor")
        clf = model.named_steps.get("classifier")
    else:
        clf = model

    PREPROCESSOR = pre

    # Prepare background data from repository if available
    base = Path(__file__).resolve().parent.parent
    data_path = base / "data" / "credit_risk_dataset.csv"
    background_X = None
    try:
        if data_path.exists() and pre is not None:
            df = pd.read_csv(data_path)
            if "loan_status" in df.columns:
                df = df.drop(columns=["loan_status"])
            # take small sample
            bg = df.sample(n=min(200, len(df)), random_state=1)
            background_X = pre.transform(bg)
            # if sparse
            try:
                background_X = background_X.toarray()
            except Exception:
                pass
            # feature names from preprocessor
            try:
                FEATURE_NAMES = list(pre.get_feature_names_out())
            except Exception:
                FEATURE_NAMES = list(df.columns)
        else:
            # no preprocessor or data available: attempt to use raw data
            if data_path.exists():
                df = pd.read_csv(data_path)
                if "loan_status" in df.columns:
                    df = df.drop(columns=["loan_status"])
                bg = df.sample(n=min(200, len(df)), random_state=1)
                background_X = bg
                FEATURE_NAMES = list(bg.columns)
            else:
                FEATURE_NAMES = None
    except Exception:
        background_X = None

    # Build TreeExplainer if classifier is tree-based
    try:
        if clf is not None:
            if background_X is not None:
                EXPLAINER = shap.TreeExplainer(clf, data=background_X)
            else:
                EXPLAINER = shap.TreeExplainer(clf)
        else:
            EXPLAINER = None
    except Exception:
        EXPLAINER = None


def explain_prediction(record: Dict[str, Any], pred_class: int) -> Dict[str, list]:
    """Return top positive and negative contributing features for prediction."""
    global EXPLAINER, FEATURE_NAMES, PREPROCESSOR
    if EXPLAINER is None:
        return {"top_positive_factors": [], "top_negative_factors": []}

    df = pd.DataFrame([record])
    # transform
    if PREPROCESSOR is not None:
        x_trans = PREPROCESSOR.transform(df)
        try:
            x_arr = x_trans.toarray()
        except Exception:
            x_arr = x_trans
    else:
        x_arr = df.values

    # compute shap values
    try:
        shap_vals = EXPLAINER.shap_values(x_arr)
    except Exception:
        return {"top_positive_factors": [], "top_negative_factors": []}

    # get class-specific shap importance
    if isinstance(shap_vals, list):
        importance = shap_vals[pred_class][0]
    else:
        arr = shap_vals
        if arr.ndim == 3:
            importance = arr[0, pred_class, :]
        else:
            importance = arr[0]

    fnames = FEATURE_NAMES if FEATURE_NAMES is not None else [f"f{i}" for i in range(len(importance))]

    import numpy as _np

    idxs = _np.argsort(_np.abs(importance))[-3:][::-1]

    top_features = []
    for i in idxs:
        top_features.append({
            "feature": fnames[i],
            "impact": float(importance[i])
        })

    # split into positive and negative top contributors
    positive = [f for f in top_features if f["impact"] > 0]
    negative = [f for f in top_features if f["impact"] < 0]

    return {"top_positive_factors": positive, "top_negative_factors": negative}


def class_to_category(cls: int) -> str:
    return {0: "Low", 1: "Medium", 2: "High"}.get(cls, "Unknown")


def map_class_to_credit_score(pred_class: int, confidence: float) -> int:
    """Deterministic credit score generation based on risk class and confidence.

    Uses the formula suggested: higher confidence increases score for low/medium
    risk and decreases score for high risk.
    """
    conf = min(max(confidence, 0.0), 1.0)
    if pred_class == 0:
        base = 750
        return int(base + conf * 150)
    elif pred_class == 1:
        base = 600
        return int(base + conf * 149)
    else:
        base = 300
        return int(base + conf * 299)


@app.on_event("startup")
def startup_load_model() -> None:
    global MODEL
    try:
        MODEL = load_trained_model()
        # initialize SHAP explainer (optional)
        try:
            build_shap_explainer(MODEL)
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(f"Failed to load trained model: {e}")


@app.post("/predict")
def predict(payload: RiskInput) -> Dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    record = payload.dict()
    try:
        pred_class, confidence = predict_record(MODEL, record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    credit_score = map_class_to_credit_score(pred_class, confidence)
    risk_category = class_to_category(pred_class)

    explanation = explain_prediction(record, pred_class)

    return {
        "credit_score": credit_score,
        "risk_category": risk_category,
        "confidence": round(confidence, 2),
        "explanation": explanation,
    }
