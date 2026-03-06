# =============================================================
# LendSight — Daily ETL + ML Credit Scoring Pipeline
# Run manually OR schedule with APScheduler
# =============================================================

from clickhouse_driver import Client
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import shap
import logging
from datetime import date

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("lendsight")

client = Client(
    host='localhost', 
    port=9000, 
    database='lendsight',
    password=''
    )

# ── FEATURES used by the ML model ────────────────────────────
FEATURES = [
    "credit_score", "debt_to_income", "employment_length",
    "loan_amount", "interest_rate", "dpd", "num_prev_defaults"
]

# ── 1. EXTRACT ───────────────────────────────────────────────
def extract_loans() -> pd.DataFrame:
    logger.info("Extracting loan data from ClickHouse...")
    rows, cols = client.execute("""
        SELECT
            l.loan_id, l.customer_id, l.product_type,
            l.loan_amount, l.outstanding_balance,
            l.interest_rate, l.dpd, l.credit_score,
            l.debt_to_income, l.is_charged_off,
            l.risk_band, l.origination_date,
            c.employment_length_yrs   AS employment_length,
            c.num_prev_defaults,
            c.annual_income
        FROM fact_loans l
        JOIN dim_customer c ON l.customer_id = c.customer_id
    """, with_column_types=True)

    col_names = [c[0] for c in cols]
    df = pd.DataFrame(rows, columns=col_names)
    logger.info(f"  Extracted {len(df):,} loan records")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")
    df = df.copy()

    # Utilisation ratio
    df["utilisation_ratio"] = (
        df["outstanding_balance"] / df["loan_amount"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    # Income-to-loan ratio
    df["income_to_loan"] = (
        df["annual_income"] / df["loan_amount"].replace(0, np.nan)
    ).fillna(0)

    # DPD bucket (ordinal encoding)
    df["dpd_bucket"] = pd.cut(
        df["dpd"],
        bins=[-1, 0, 29, 89, float("inf")],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # High-risk flag
    df["is_high_risk"] = (
        (df["debt_to_income"] > 0.45) |
        (df["credit_score"]   < 580)  |
        (df["num_prev_defaults"] > 1)
    ).astype(int)

    return df.dropna(subset=FEATURES)

# ── 3. TRAIN ML MODEL ────────────────────────────────────────
def train_model(df: pd.DataFrame):
    logger.info("Training credit scoring model...")
    X = df[FEATURES]
    y = df["is_charged_off"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"  Model AUC: {auc:.4f}")
    logger.info("\n" + classification_report(y_test, model.predict(X_test)))

    return model, auc

# ── 4. SHAP FEATURE IMPORTANCE ───────────────────────────────
def get_shap_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating SHAP values...")
    explainer  = shap.TreeExplainer(model.named_steps["clf"])
    X_scaled   = model.named_steps["scaler"].transform(X_sample)
    shap_vals  = explainer.shap_values(X_scaled)
    importance = pd.DataFrame({
        "feature"   : FEATURES,
        "importance": np.abs(shap_vals).mean(axis=0)
    }).sort_values("importance", ascending=False)

    logger.info("  Top features:\n" + importance.to_string(index=False))
    return importance

# ── 5. SCORE ALL LOANS & SAVE ────────────────────────────────
def score_and_save(df: pd.DataFrame, model):
    logger.info("Scoring all loans...")
    df["ml_default_score"] = model.predict_proba(df[FEATURES])[:, 1].round(4)
    df["score_date"]       = date.today()
    df["score_band"]       = pd.cut(
        df["ml_default_score"],
        bins=[0, 0.10, 0.25, 0.50, 1.0],
        labels=["Low Risk","Medium Risk","High Risk","Very High Risk"],
        include_lowest=True
    ).astype(str).fillna("Unknown")

    # Clear today's scores and reload (idempotent)
    client.execute("ALTER TABLE fact_ml_scores DELETE WHERE score_date = today()")

    df["risk_band"] = df["risk_band"].fillna("")  # Ensure no nulls in risk_band

    rows = df[["loan_id","credit_score","risk_band",
               "ml_default_score","score_date","score_band"]].to_dict("records")
    # debug check for non-string in string columns
    for i,row in enumerate(rows[:20]):
        for col in ["loan_id","risk_band","score_band"]:
            val = row[col]
            if val is not None and not isinstance(val, str):
                logger.error(f"Row {i} col {col} has type {type(val)} value {val!r}")
    # also check overall
    for col in ["loan_id","risk_band","score_band"]:
        types = set(type(r[col]) for r in rows)
        logger.info(f"Column {col} types: {types}")

    client.execute("INSERT INTO fact_ml_scores VALUES", rows)
    logger.info(f"  Saved {len(rows):,} scores to fact_ml_scores")

# ── MAIN ─────────────────────────────────────────────────────
def run_pipeline():
    logger.info("═══ LendSight Daily Pipeline Starting ═══")
    raw      = extract_loans()
    df       = engineer_features(raw)
    model, _ = train_model(df)
    get_shap_importance(model, df[FEATURES].sample(min(1000, len(df))))
    score_and_save(df, model)
    logger.info("═══ Pipeline Complete ✓ ═══")

if __name__ == "__main__":
    run_pipeline()