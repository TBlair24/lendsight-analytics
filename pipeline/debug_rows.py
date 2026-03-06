import pandas as pd, numpy as np
from clickhouse_driver import Client
client=Client(host='localhost',port=9000,database='lendsight',password='')
rows,cols=client.execute("""
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
col_names=[c[0] for c in cols]
df=pd.DataFrame(rows,columns=col_names)
# engineer features like pipeline
from datetime import date

# feature engineering not needed to check risk_band
df['risk_band']=df['risk_band'].fillna('')
df['ml_default_score']=0.5

df['score_date']=date.today()
df['score_band']=pd.cut(df['ml_default_score'],bins=[0,0.10,0.25,0.50,1.0],labels=['Low Risk','Medium Risk','High Risk','Very High Risk']).astype(str)

rows=df[['loan_id','credit_score','risk_band','ml_default_score','score_date','score_band']].to_dict('records')
bad=[]
for r in rows:
    for k,v in r.items():
        if k in ['loan_id','risk_band','score_band'] and not isinstance(v,str):
            bad.append((k,v))
            break
print('bad examples', bad[:20])
# print unique types per column
for col in ['loan_id','risk_band','score_band']:
    types=set(type(r[col]) for r in rows)
    print(col, types)
