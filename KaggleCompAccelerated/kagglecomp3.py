import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# REMOVED: negative value replacement — -1, -2, -3, -4 are REAL data values in this dataset
# Replacing them with NaN destroys real signal and hurts AUC

y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

# ==========================================
# 2. Identify & Format Categorical Features
# ==========================================
numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()

# No fillna needed since we're not creating NaNs anymore
for col in categorical_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"Detected {len(categorical_features)} categoricals and {len(numeric_features)} continuous floats.")

# ==========================================
# 3. Feature Engineering — Top Pairwise Interactions
# ==========================================
print("Running quick baseline to find top features for interactions...")
quick_model = CatBoostClassifier(iterations=200, random_seed=42, verbose=False)
quick_model.fit(X, y, cat_features=categorical_features)

importances = pd.Series(quick_model.get_feature_importance(), index=X.columns)
top_features = importances.nlargest(8).index.tolist()
print(f"Top 8 features: {top_features}")

# Only create interactions between continuous top features
for i, col1 in enumerate(top_features):
    for col2 in top_features[i+1:]:
        if col1 in numeric_features and col2 in numeric_features:
            X[f"{col1}_x_{col2}"]       = X[col1] * X[col2]
            X_test[f"{col1}_x_{col2}"]  = X_test[col1] * X_test[col2]
            X[f"{col1}_div_{col2}"]      = X[col1] / (X[col2] + 1e-6)
            X_test[f"{col1}_div_{col2}"] = X_test[col1] / (X_test[col2] + 1e-6)

print(f"Features after interaction engineering: {X.shape[1]}")

# ==========================================
# 4. Define the Optuna Objective for CatBoost
# ==========================================
def objective(trial):
    cb_params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 15.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count': trial.suggest_int('border_count', 64, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }

    cb_model = CatBoostClassifier(**cb_params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cb_model.fit(
            X_tr, y_tr,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )
        aucs.append(cb_model.get_best_score()['validation']['AUC'])

    return np.mean(aucs)

# ==========================================
# 5. Execute Optuna Optimization
# ==========================================
print("Starting CatBoost Optuna Optimization...")
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)  # kills bad trials early
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n==========================================")
print(f"Best Trial AUC: {study.best_trial.value:.5f}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
print("==========================================\n")

# ==========================================
# 6. K-Fold Prediction Averaging for Submission
# ==========================================
print("Executing K-Fold Prediction Averaging for final submission...")

test_preds_sum = np.zeros(len(X_test))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_idx, val_idx in cv.split(X, y):
    print(f"Training Final Model Fold {fold}/5...")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = CatBoostClassifier(
        **study.best_trial.params,
        eval_metric='AUC',
        random_seed=42 + fold,  # slight seed diversity per fold
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

    test_preds_sum += fold_model.predict_proba(X_test)[:, 1]
    fold += 1

final_test_predictions = test_preds_sum / 5.0

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': final_test_predictions
})

submission_df.to_csv('catboost_kfold_submission.csv', index=False)
print("Finished! Saved K-Fold averaged predictions to 'catboost_kfold_submission.csv'.")