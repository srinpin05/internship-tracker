import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data (Keeping -1s as signal!)
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# DO NOT uncomment the missing_placeholders. We want CatBoost to see them!

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
# 3. Multi-Seed Self-Stacking Engine
# ==========================================
# We will run the 5-fold process 3 times with these seeds
seeds = [42, 420, 4200]

# Arrays to hold the massive stacked predictions
final_test_preds = np.zeros(len(X_test))
global_oof_aucs = []

print(f"Starting Multi-Seed Stack with {len(seeds)} different initializations...")

for seed in seeds:
    print(f"\n--- Initiating Seed {seed} ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    seed_test_preds = np.zeros(len(X_test))
    fold_aucs = []
    
    fold = 1
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Your elite Trial 44 Hyperparameters
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
            random_seed=seed + fold, # Total randomness per model
            verbose=False,
            thread_count=-1
        )
        
        model.fit(
            X_tr, y_tr,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100, 
            verbose=False
        )
        
        # Track Local Validation
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.5f}")
        
        # Track Test Predictions for this seed
        seed_test_preds += model.predict_proba(X_test)[:, 1] / 5.0
        fold += 1
        
    seed_avg_auc = np.mean(fold_aucs)
    global_oof_aucs.append(seed_avg_auc)
    print(f"-> Seed {seed} Average CV AUC: {seed_avg_auc:.5f}")
    
    # Add this seed's final test predictions to our master stacked array
    final_test_preds += seed_test_preds / len(seeds)

# ==========================================
# 4. Final Scoring & Submission
# ==========================================
print("\n==========================================")
print(f"🏆 ULTIMATE STACKED CV AUC SCORE: {np.mean(global_oof_aucs):.5f}")
print("==========================================\n")

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': final_test_preds
})

submission_df.to_csv('catboost_multiseed_stack_submission.csv', index=False)
print("Finished! Saved 'catboost_multiseed_stack_submission.csv'.")