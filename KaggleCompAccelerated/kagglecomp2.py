import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load & Basic Cleaning
# ==========================================
print("Loading Data...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# Clean Nulls (Keeping it simple to avoid noise)
missing_placeholders = [-1, -2, -3, -4]
train_df.replace(missing_placeholders, np.nan, inplace=True)
test_df.replace(missing_placeholders, np.nan, inplace=True)

y = train_df['Y']
X = train_df.drop(['Id', 'Y'], axis=1)
X_test = test_df.drop(['Id'], axis=1)
test_ids = test_df['Id']

# Identify Categoricals
categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()
for col in categorical_features:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

# ==========================================
# 2. Part A: Find Best Parameters (Optuna)
# ==========================================
def objective(trial):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1 # Mac Performance
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, cat_features=categorical_features, 
                  eval_set=(X_val, y_val), early_stopping_rounds=50)
        aucs.append(model.get_best_score()['validation']['AUC'])
        
    return np.mean(aucs)

print("Starting Optuna search for best CatBoost params...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) # Increase to 50 if you have time

best_params = study.best_params
best_params.update({'iterations': 2000, 'eval_metric': 'AUC', 'random_seed': 42, 'thread_count': -1})

# ==========================================
# 3. Part B: Stage 1 Training (Initial Predictions)
# ==========================================
print("\nStage 1: Training initial model to generate Pseudo-Labels...")
model_stage1 = CatBoostClassifier(**best_params, verbose=0)
model_stage1.fit(X, y, cat_features=categorical_features)

# Predict probabilities on test set
test_probs = model_stage1.predict_proba(X_test)[:, 1]

# ==========================================
# 4. Part C: Stage 2 - PSEUDO-LABELING (The 0.92 Booster)
# ==========================================
# We pick the test samples the model is MOST confident about (Top 5% 0s and 1s)
CONFIDENCE_THRESHOLD = 0.05 
n_top = int(len(X_test) * CONFIDENCE_THRESHOLD)

# Get indices for highest and lowest probabilities
pseudo_1_idx = np.argsort(test_probs)[-n_top:]
pseudo_0_idx = np.argsort(test_probs)[:n_top]

# Create a Pseudo-labeled dataset
X_pseudo = pd.concat([X_test.iloc[pseudo_1_idx], X_test.iloc[pseudo_0_idx]])
y_pseudo = pd.Series([1]*n_top + [0]*n_top)

# Combine with original training data
X_combined = pd.concat([X, X_pseudo], axis=0)
y_combined = pd.concat([y, y_pseudo], axis=0)

print(f"Added {len(y_pseudo)} Pseudo-labels from Test Data.")

# ==========================================
# 5. Final Training (10-Fold CV for maximum stability)
# ==========================================
print("\nStage 2: Final 10-Fold CV Training on Combined Data...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
final_test_preds = np.zeros(len(X_test))

for i, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
    X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
    y_tr, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
    
    final_model = CatBoostClassifier(**best_params, verbose=0)
    final_model.fit(X_tr, y_tr, cat_features=categorical_features, 
                    eval_set=(X_val, y_val), early_stopping_rounds=100)
    
    final_test_preds += final_model.predict_proba(X_test)[:, 1] / 10
    print(f"Fold {i+1} complete.")

# Save
pd.DataFrame({'Id': test_ids, 'Y': final_test_preds}).to_csv('pseudo_label_catboost.csv', index=False)
print("\nSuccess! Pseudo-labeled CatBoost submission saved.")