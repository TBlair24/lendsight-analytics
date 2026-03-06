-- ============================================================
-- LendSight Analytics — ClickHouse Schema
-- Run this ONCE to create all tables
-- ============================================================

USE lendsight;

-- Dimension: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date            Date,
    year            UInt16,
    quarter         UInt8,
    month           UInt8,
    month_name      String,
    week            UInt8,
    day_of_week     UInt8,
    is_weekend      UInt8,
    month_start     Date,
    quarter_label   String,
    fy_year         UInt16
) ENGINE = MergeTree()
ORDER BY date;

-- Dimension: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id             String,
    first_name              String,
    last_name               String,
    date_of_birth           Date,
    age                     UInt8,
    region                  String,
    employment_status       String,
    employment_length_yrs   Float32,
    annual_income           Float64,
    risk_band               String,
    credit_score            UInt16,
    num_prev_defaults       UInt8,
    created_date            Date
) ENGINE = MergeTree()
ORDER BY customer_id;

-- Fact: Loans
CREATE TABLE IF NOT EXISTS fact_loans (
    loan_id                 String,
    customer_id             String,
    product_type            String,
    origination_date        Date,
    term_months             UInt16,
    loan_amount             Float64,
    outstanding_balance     Float64,
    interest_rate           Float32,
    monthly_payment         Float64,
    dpd                     UInt16,
    risk_band               String,
    credit_score            UInt16,
    debt_to_income          Float32,
    employment_length       Float32,
    region                  String,
    is_charged_off          UInt8,
    snapshot_date           Date,
    origination_month       Date
) ENGINE = MergeTree()
ORDER BY (snapshot_date, loan_id);

-- Fact: Payments
CREATE TABLE IF NOT EXISTS fact_payments (
    payment_id          String,
    loan_id             String,
    payment_date        Date,
    amount              Float64,
    interest_portion    Float64,
    principal_portion   Float64,
    payment_type        String,
    status              String
) ENGINE = MergeTree()
ORDER BY (payment_date, loan_id);

-- Fact: ML Scores
CREATE TABLE IF NOT EXISTS fact_ml_scores (
    loan_id             String,
    credit_score        UInt16,
    risk_band           String,
    ml_default_score    Float32,
    score_date          Date,
    score_band          String
) ENGINE = MergeTree()
ORDER BY (score_date, loan_id);