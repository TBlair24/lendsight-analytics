# =============================================================
# LendSight Analytics — Synthetic Data Generator
# Generates ~50,000 realistic loan records across 3 years
# =============================================================

import pandas as pd 
import numpy as np
from faker import Faker
from datetime import date, timedelta
import os

fake = Faker('en_GB')
np.random.seed(42)
Faker.seed(42)

OUTPUT_DIR = r"C:\Projects\lendsight-analytics\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating LendSight synthetic data...")

# ── PARAMETERS ──────────────────────────────────────────────
N_CUSTOMERS    = 15_000
N_LOANS        = 50_000
START_DATE     = date(2022, 1, 1)
END_DATE       = date(2025, 12, 31)

PRODUCTS = {
    "Personal Loan" : {"weight": 0.40, "rate_range": (0.08, 0.28),  "amount_range": (1_000,  35_000), "term_months_range": (12, 60)},
    "Auto Finance"  : {"weight": 0.25, "rate_range": (0.06, 0.18),  "amount_range": (5_000,  45_000), "term_months_range": (24, 72)},
    "Credit Card"   : {"weight": 0.20, "rate_range": (0.18, 0.39),  "amount_range": (500,    15_000), "term_months_range": (12, 36)},
    "Mortgage"      : {"weight": 0.15, "rate_range": (0.035, 0.065),"amount_range": (80_000, 500_000),"term_months_range": (120,360)},
}

REGIONS = ["London", "Manchester", "Birmingham", "Leeds", "Bristol",
           "Edinburgh", "Cardiff", "Glasgow", "Liverpool", "Sheffield"]

RISK_BANDS = {
    "Prime"        : {"score_range": (750, 850), "default_prob": 0.005},
    "Near-Prime"   : {"score_range": (680, 749), "default_prob": 0.025},
    "Sub-Prime"    : {"score_range": (580, 679), "default_prob": 0.08},
    "Deep Sub-Prime": {"score_range": (400, 579), "default_prob": 0.18},
}

# ── 1. DIM_CUSTOMER ─────────────────────────────────────────
print("  → Generating dim_customer...")

def assign_risk_band():
    return np.random.choice(
        list(RISK_BANDS.keys()),
        p=[0.35, 0.35, 0.20, 0.10]
    )


customers = []
for i in range(N_CUSTOMERS):
    rb    = assign_risk_band()
    score = np.random.randint(*RISK_BANDS[rb]["score_range"])
    dob   = fake.date_of_birth(minimum_age=21, maximum_age=72)
    customers.append({
        "customer_id"       : f"CUST{i+1:06d}",
        "first_name"        : fake.first_name(),
        "last_name"         : fake.last_name(),
        "date_of_birth"     : dob,
        "age"               : (date.today() - dob).days // 365,
        "region"            : np.random.choice(REGIONS),
        "employment_status" : np.random.choice(
            ["Employed","Self-Employed","Contractor","Retired"],
            p=[0.65, 0.18, 0.10, 0.07]
        ),
        "employment_length_yrs" : round(np.random.exponential(5), 1),
        "annual_income"     : round(np.random.lognormal(10.5, 0.5), -2),
        "risk_band"         : rb,
        "credit_score"      : score,
        "num_prev_defaults" : np.random.choice([0,1,2,3], p=[0.75,0.15,0.07,0.03]),
        "created_date"      : fake.date_between(START_DATE, END_DATE),
    })

dim_customer = pd.DataFrame(customers)

# ── 2. FACT_LOANS ────────────────────────────────────────────
print("  → Generating fact_loans...")

product_names = list(PRODUCTS.keys())
product_weights = [PRODUCTS[p]["weight"] for p in product_names]

loans = []
for i in range(N_LOANS):
    cust       = dim_customer.sample(1).iloc[0]
    product    = np.random.choice(product_names, p=product_weights)
    p          = PRODUCTS[product]
    rb         = cust["risk_band"]
    def_prob   = RISK_BANDS[rb]["default_prob"]

    orig_date  = fake.date_between(START_DATE, END_DATE)
    amount     = round(np.random.uniform(*p["amount_range"]), -2)
    rate       = round(np.random.uniform(*p["rate_range"]), 4)
    term       = np.random.randint(*p["term_months_range"])
    dti        = round(np.clip(np.random.normal(0.32, 0.12), 0.05, 0.75), 3)

    # Simulate DPD based on risk profile
    is_default = np.random.random() < def_prob
    if is_default:
        dpd = int(np.random.choice([30,60,90,120,180], p=[0.3,0.25,0.2,0.15,0.1]))
    else:
        dpd = int(np.random.choice([0,0,0,0,0,1,5,10,15,25], p=[0.65,0.1,0.05,0.05,0.05,0.03,0.03,0.02,0.01,0.01]))

    months_on_book = (date.today() - orig_date).days // 30
    months_elapsed = min(months_on_book, term)
    monthly_rate   = rate / 12
    if monthly_rate > 0:
        factor = (1 + monthly_rate)**term
        monthly_payment = amount * monthly_rate * factor / (factor - 1)
    else:
        monthly_payment = amount / term

    outstanding = max(0, amount - (monthly_payment * months_elapsed * (1 - def_prob * 0.3)))
    is_written_off = dpd >= 180

    loans.append({
        "loan_id"              : f"LN{i+1:07d}",
        "customer_id"          : cust["customer_id"],
        "product_type"         : product,
        "origination_date"     : orig_date,
        "term_months"          : term,
        "loan_amount"          : amount,
        "outstanding_balance"  : round(outstanding, 2),
        "interest_rate"        : rate,
        "monthly_payment"      : round(monthly_payment, 2),
        "dpd"                  : dpd,
        "risk_band"            : rb,
        "credit_score"         : cust["credit_score"],
        "debt_to_income"       : dti,
        "employment_length"    : cust["employment_length_yrs"],
        "region"               : cust["region"],
        "is_charged_off"       : int(is_written_off),
        "snapshot_date"        : date.today(),
        "origination_month"    : orig_date.replace(day=1),
    })

fact_loans = pd.DataFrame(loans)

# ── 3. FACT_PAYMENTS ─────────────────────────────────────────
print("  → Generating fact_payments...")

payments = []
for _, loan in fact_loans.sample(min(30_000, len(fact_loans))).iterrows():
    months = min((date.today() - loan["origination_date"]).days // 30, loan["term_months"])
    for m in range(1, months + 1):
        pay_date   = loan["origination_date"] + timedelta(days=30*m)
        missed     = loan["is_charged_off"] and m > (months - 3)
        if not missed:
            payments.append({
                "payment_id"       : f"PAY{len(payments)+1:09d}",
                "loan_id"          : loan["loan_id"],
                "payment_date"     : pay_date,
                "amount"           : loan["monthly_payment"],
                "interest_portion" : round(loan["outstanding_balance"] * (loan["interest_rate"]/12), 2),
                "principal_portion": round(loan["monthly_payment"] - loan["outstanding_balance"] * (loan["interest_rate"]/12), 2),
                "payment_type"     : "scheduled",
                "status"           : "completed",
            })

fact_payments = pd.DataFrame(payments)

# ── 4. DIM_DATE ──────────────────────────────────────────────
print("  → Generating dim_date...")
date_range = pd.date_range(START_DATE, END_DATE + timedelta(days=365), freq='D')
dim_date = pd.DataFrame({
    "date"           : date_range,
    "year"           : date_range.year,
    "quarter"        : date_range.quarter,
    "month"          : date_range.month,
    "month_name"     : date_range.strftime('%B'),
    "week"           : date_range.isocalendar().week.values,
    "day_of_week"    : date_range.dayofweek,
    "is_weekend"     : date_range.dayofweek >= 5,
    "month_start"    : date_range.to_period('M').to_timestamp(),
    "quarter_label"  : 'Q' + date_range.quarter.astype(str) + ' ' + date_range.year.astype(str),
    "fy_year"        : np.where(date_range.month >= 4, date_range.year + 1, date_range.year),
})

# ── 5. FACT_ML_SCORES (placeholder) ─────────────────────────
fact_ml_scores = fact_loans[["loan_id", "credit_score", "risk_band"]].copy()
fact_ml_scores["ml_default_score"] = np.random.beta(2, 8, len(fact_ml_scores)).round(4)
fact_ml_scores["score_date"]       = date.today()
fact_ml_scores["score_band"] = pd.cut(
    fact_ml_scores["ml_default_score"],
    bins=[0, 0.1, 0.25, 0.5, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
)

# ── 6. SAVE TO CSV ───────────────────────────────────────────
print("  → Saving CSV files...")

dim_customer.to_csv(f"{OUTPUT_DIR}/dim_customer.csv",    index=False)
fact_loans.to_csv(  f"{OUTPUT_DIR}/fact_loans.csv",      index=False)
fact_payments.to_csv(f"{OUTPUT_DIR}/fact_payments.csv",  index=False)
dim_date.to_csv(    f"{OUTPUT_DIR}/dim_date.csv",        index=False)
fact_ml_scores.to_csv(f"{OUTPUT_DIR}/fact_ml_scores.csv",index=False)

print(f"""
✅ Data generation complete!
   dim_customer  : {len(dim_customer):,} rows
   fact_loans    : {len(fact_loans):,} rows
   fact_payments : {len(fact_payments):,} rows
   dim_date      : {len(dim_date):,} rows
   fact_ml_scores: {len(fact_ml_scores):,} rows

   Files saved to: {OUTPUT_DIR}
""")

