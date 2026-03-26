import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Original Data & Winning Submission
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')
best_sub = pd.read_csv('catboost_multiseed_stack_submission.csv')

y_orig = train_df['Y']
X_orig = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

# ==========================================
# 2. Extract Pseudo-Labels
# ==========================================
print("Extracting high-confidence pseudo-labels...")
pseudo_test = test_df.drop(columns=['Id']).copy()
pseudo_test['Y'] = best_sub['Y']

# Top 1% and Bottom 1% confidence bounds
confident_zeros = pseudo_test[pseudo_test['Y'] < 0.01].copy()
confident_ones = pseudo_test[pseudo_test['Y'] > 0.99].copy()

# Hardcode the labels
confident_zeros['Y'] = 0.0
confident_ones['Y'] = 1.0

pseudo_labels_df = pd.concat([confident_zeros, confident_ones])
pseudo_y = pseudo_labels_df['Y']
pseudo_X = pseudo_labels_df.drop(columns=['Y'])

print(f"  -> Found {len(confident_zeros)} confident zeros and {len(confident_ones)} confident ones.")
print(f"  -> Ready to inject {len(pseudo_labels_df)} bonus rows into the training folds.")

# ==========================================
# 3. Format Categorical Features
# ==========================================
categorical_features = X_orig.select_dtypes(exclude=['float64']).columns.tolist()

for col in categorical_features:
    X_orig[col] = X_orig[col].fillna('Missing_Category').astype(str)
    pseudo_X[col] = pseudo_X[col].fillna('Missing_Category').astype(str)
    X_test[col] = X_test[col].fillna('Missing_Category').astype(str)

# ==========================================
# 4. Pure Validation Training Loop
# ==========================================
print("Executing 5-Fold Training with Pure Validation...")
test_preds_sum = np.zeros(len(X_test))
oof_preds = np.zeros(len(X_orig))
fold_aucs = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_idx, val_idx in cv.split(X_orig, y_orig):
    print(f"  Training Fold {fold}/5...")
    
    # 1. Get the pure training data for this fold
    X_tr_pure, X_val = X_orig.iloc[train_idx], X_orig.iloc[val_idx]
    y_tr_pure, y_val = y_orig.iloc[train_idx], y_orig.iloc[val_idx]
    
    # 2. Inject the pseudo-labels ONLY into the training data
    X_tr_aug = pd.concat([X_tr_pure, pseudo_X]).reset_index(drop=True)
    y_tr_aug = pd.concat([y_tr_pure, pseudo_y]).reset_index(drop=True)
    
    # Elite Trial 44 Hyperparameters
    model = CatBoostClassifier(
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
    
    # Train on Augmented, Validate on Pure
    model.fit(
        X_tr_aug, y_tr_aug,
        cat_features=categorical_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100, 
        verbose=False
    )
    
    # Score the pure validation fold
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    fold_auc = roc_auc_score(y_val, val_preds)
    fold_aucs.append(fold_auc)
    print(f"    -> Pure Fold {fold} AUC: {fold_auc:.5f}")
    
    test_preds_sum += model.predict_proba(X_test)[:, 1] / 5.0
    fold += 1

# ==========================================
# 5. Final Scoring
# ==========================================
final_oof_auc = np.mean(fold_aucs)

print("\n==========================================")
print(f"🏆 PURE PSEUDO-LABELED CV AUC SCORE: {final_oof_auc:.5f}")
print("==========================================\n")

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': test_preds_sum
})

submission_df.to_csv('catboost_pseudo_labeled_submission.csv', index=False)
print("Finished! Saved 'catboost_pseudo_labeled_submission.csv'.")