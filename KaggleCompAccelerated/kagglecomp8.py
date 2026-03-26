import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

print("Loading data for Local Rank-Average Validation...")
train_df = pd.read_csv('training_data.csv')

y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])

categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()
for col in categorical_features:
    X[col] = X[col].fillna('Missing_Category').astype(str)

print("Executing 5-Fold Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to hold the validation predictions for Model A and Model B
oof_model_A = np.zeros(len(X))
oof_model_B = np.zeros(len(X))

fold = 1
for train_idx, val_idx in cv.split(X, y):
    print(f"  Training Fold {fold}/5...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Model A: Your Champion (Seed 42)
    model_A = CatBoostClassifier(
        iterations=2202, depth=6, learning_rate=0.0846, l2_leaf_reg=5.36,
        random_strength=7.16, auto_class_weights='SqrtBalanced', eval_metric='AUC',
        random_seed=42+fold, verbose=False, thread_count=-1
    )
    model_A.fit(X_tr, y_tr, cat_features=categorical_features, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_model_A[val_idx] = model_A.predict_proba(X_val)[:, 1]

    # Model B: The Backup (Different Seed to simulate your second-best model)
    model_B = CatBoostClassifier(
        iterations=2202, depth=6, learning_rate=0.0846, l2_leaf_reg=5.36,
        random_strength=7.16, auto_class_weights='SqrtBalanced', eval_metric='AUC',
        random_seed=420+fold, verbose=False, thread_count=-1
    )
    model_B.fit(X_tr, y_tr, cat_features=categorical_features, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
    oof_model_B[val_idx] = model_B.predict_proba(X_val)[:, 1]
    
    fold += 1

# ==========================================
# Scoring the Individual Models
# ==========================================
auc_A = roc_auc_score(y, oof_model_A)
auc_B = roc_auc_score(y, oof_model_B)
print("\n==========================================")
print(f"Model A (Champion) OOF AUC: {auc_A:.5f}")
print(f"Model B (Backup) OOF AUC:   {auc_B:.5f}")

# ==========================================
# The Rank Average Magic
# ==========================================
print("\nCalculating Rank Average...")
# Convert probabilities to ranks
ranks_A = rankdata(oof_model_A)
ranks_B = rankdata(oof_model_B)

# Blend the ranks (75% Champion / 25% Backup)
blended_ranks = (ranks_A * 0.75) + (ranks_B * 0.25)

# Score the blended ranks! 
final_rank_auc = roc_auc_score(y, blended_ranks)

print("==========================================")
print(f"🏆 RANK AVERAGED OOF AUC: {final_rank_auc:.5f}")
print("==========================================\n")