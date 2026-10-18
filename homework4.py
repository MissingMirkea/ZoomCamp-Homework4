import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# Step 0 & 1: Load and Clean Data
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
df = pd.read_csv(url)

cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if 'converted' in num_cols:
    num_cols.remove('converted')
if 'converted' in cat_cols:
    cat_cols.remove('converted')

# Fill missing values
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

# ---------------------------------------------------------
# Split Data with random_state=1
# ---------------------------------------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_full_train = df_full_train['converted'].values

# ---------------------------------------------------------
# Q1: ROC AUC Feature Importance
# ---------------------------------------------------------
num_candidates = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']
auc_scores = {}

for col in num_candidates:
    score = roc_auc_score(y_train, df_train[col])
    if score < 0.5:
        score = roc_auc_score(y_train, -df_train[col])
    auc_scores[col] = score

best_num_feat = max(auc_scores, key=auc_scores.get)
print(f"--- Q1 ---")
for k, v in auc_scores.items():
    print(f"{k}: {v:.4f}")
print(f">> Highest AUC variable: {best_num_feat}\n")

# ---------------------------------------------------------
# Q2: Logistic Regression Baseline AUC
# ---------------------------------------------------------
all_feats = cat_cols + num_cols

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[all_feats].to_dict(orient='records'))
X_val = dv.transform(df_val[all_feats].to_dict(orient='records'))

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred_val = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred_val)

print(f"--- Q2 ---")
print(f"Validation AUC: {val_auc:.4f} -> rounded: {round(val_auc, 3)}\n")

# ---------------------------------------------------------
# Q3 & Q4: Precision, Recall, and F1 Curves
# ---------------------------------------------------------
thresholds = np.linspace(0.0, 1.0, 101)
p_scores, r_scores, f1_scores = [], [], []

min_diff = float('inf')
best_pr_thresh = None

best_f1 = -1
best_f1_thresh = None

for t in thresholds:
    y_pred_bin = (y_pred_val >= t).astype(int)
    p = precision_score(y_val, y_pred_bin, zero_division=0)
    r = recall_score(y_val, y_pred_bin, zero_division=0)
    
    if p + r > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0.0
        
    p_scores.append(p)
    r_scores.append(r)
    f1_scores.append(f1)
    
    # Check PR intersection
    if p > 0 and r > 0:
        diff = abs(p - r)
        if diff < min_diff:
            min_diff = diff
            best_pr_thresh = round(t, 2)
            
    # Check max F1
    if f1 > best_f1:
        best_f1 = f1
        best_f1_thresh = round(t, 2)

print(f"--- Q3 ---")
print(f">> PR Intersection Threshold: {best_pr_thresh}\n")

print(f"--- Q4 ---")
print(f">> Max F1 Score ({best_f1:.4f}) Threshold: {best_f1_thresh}\n")

# ---------------------------------------------------------
# Q5: 5-Fold Cross Validation
# ---------------------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_tr = df_full_train.iloc[train_idx]
    df_va = df_full_train.iloc[val_idx]
    
    y_tr = df_tr['converted'].values
    y_va = df_va['converted'].values
    
    dv_cv = DictVectorizer(sparse=False)
    X_tr = dv_cv.fit_transform(df_tr[all_feats].to_dict(orient='records'))
    X_va = dv_cv.transform(df_va[all_feats].to_dict(orient='records'))
    
    m_cv = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    m_cv.fit(X_tr, y_tr)
    
    y_pred = m_cv.predict_proba(X_va)[:, 1]
    scores.append(roc_auc_score(y_va, y_pred))

std_cv = np.std(scores)
print(f"--- Q5 ---")
print(f"5-Fold AUC Mean: {np.mean(scores):.4f}, Std: {std_cv:.4f} -> rounded: {round(std_cv, 3)}\n")

# ---------------------------------------------------------
# Q6: Tuning C Parameter
# ---------------------------------------------------------
c_values = [0.000001, 0.001, 1]
c_summary = {}

print(f"--- Q6 ---")
for C in c_values:
    c_scores_list = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_va = df_full_train.iloc[val_idx]
        
        y_tr = df_tr['converted'].values
        y_va = df_va['converted'].values
        
        dv_cv = DictVectorizer(sparse=False)
        X_tr = dv_cv.fit_transform(df_tr[all_feats].to_dict(orient='records'))
        X_va = dv_cv.transform(df_va[all_feats].to_dict(orient='records'))
        
        m_c = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        m_c.fit(X_tr, y_tr)
        
        y_pred = m_c.predict_proba(X_va)[:, 1]
        c_scores_list.append(roc_auc_score(y_va, y_pred))
        
    mean_auc = round(np.mean(c_scores_list), 3)
    std_auc = round(np.std(c_scores_list), 3)
    c_summary[C] = (mean_auc, std_auc)
    print(f"C={C:<8} | Mean AUC: {mean_auc:.3f} | Std: {std_auc:.3f}")

best_c = max(c_values, key=lambda c: (c_summary[c][0], -c_summary[c][1], -c))
print(f">> Best C value: {best_c}")