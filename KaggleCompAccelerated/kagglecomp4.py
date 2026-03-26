import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data & Clean Hidden Nulls
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# Crucial for CatBoost to handle floats properly
#missing_placeholders = [-1, -2, -3, -4]
#train_df.replace(missing_placeholders, np.nan, inplace=True)
#test_df.replace(missing_placeholders, np.nan, inplace=True)

y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

# ==========================================
# 2. Identify & Format Categorical Features
# ==========================================
categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()

for col in categorical_features:
    X[col] = X[col].fillna('Missing_Category').astype(str)
    X_test[col] = X_test[col].fillna('Missing_Category').astype(str)

# ==========================================
# 3. K-Fold Prediction Averaging (The 0.95 Engine)
# ==========================================
print("Executing 5-Fold CatBoost Averaging...")

# Array to hold the sum of predictions from all 5 folds for the TEST set
test_preds_sum = np.zeros(len(X_test))

# Array to hold the Out-Of-Fold predictions to calculate our local AUC
oof_preds = np.zeros(len(X))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_idx, val_idx in cv.split(X, y):
    print(f"  Training Fold {fold}/5...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Your exact winning Trial 44 Hyperparameters
    fold_model = CatBoostClassifier(
        iterations=2202,
        depth=6,
        learning_rate=0.08465629610185188,
        l2_leaf_reg=5.3675687836265595,
        random_strength=7.1616238589515815,
        bagging_temperature=1.6121296973145813,
        border_count=157,
        min_data_in_leaf=42,
        auto_class_weights='SqrtBalanced',
        eval_metric='AUC',
        random_seed=42 + fold, 
        verbose=False,
        thread_count=-1
    )
    
    fold_model.fit(
        X_tr, y_tr,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100, 
        verbose=False
    )
    
    # Save validation predictions for our local AUC score
    oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]
    
    # Add this fold's test predictions to the running total
    test_preds_sum += fold_model.predict_proba(X_test)[:, 1]
    fold += 1

# ==========================================
# 4. Final Scoring & Kaggle Submission
# ==========================================
# Calculate the final Local AUC score
final_oof_auc = roc_auc_score(y, oof_preds)

print("\n==========================================")
print(f"🏆 FINAL OOF AUC SCORE: {final_oof_auc:.5f}")
print("==========================================\n")

print("Averaging final probabilities...")
final_test_predictions = test_preds_sum / 5.0

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': final_test_predictions
})

submission_df.to_csv('catboost_kfold_final_push.csv', index=False)
print("Finished! Saved 'catboost_kfold_final_push.csv'.")