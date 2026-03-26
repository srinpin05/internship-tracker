import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, TargetEncoder
from xgboost import XGBClassifier
import warnings

# Suppress convergence warnings from Elastic Net's SAGA solver
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data & Clean Hidden Nulls
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# Convert negative placeholders to NaN for proper imputation
missing_placeholders = [-1, -2, -3, -4]
train_df.replace(missing_placeholders, np.nan, inplace=True)
test_df.replace(missing_placeholders, np.nan, inplace=True)

y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

# ==========================================
# 2. Upgraded Feature Engineering (Target Encoding)
# ==========================================
numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
])

# THE UPGRADE: Swap One-Hot for Target Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target_enc', TargetEncoder(target_type='binary', smooth='auto'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 3. Define the Optuna Objective (XGBoost ONLY)
# ==========================================
def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 800),
        'max_depth': trial.suggest_int('xgb_max_depth', 4, 10), # Allowed it to go slightly deeper
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-3, 10.0, log=True),
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    xgb_model = XGBClassifier(**xgb_params)
    
    # Notice: I removed the Elastic Net selector here so XGBoost gets all the data
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()

# ==========================================
# 4. Execute Optuna Optimization
# ==========================================
print("Starting Joint Optuna Optimization (Elastic Net + XGBoost)...")
study = optuna.create_study(direction='maximize')
# Let it rip for 30 trials
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n==========================================")
print(f"Best Trial AUC: {study.best_trial.value:.5f}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
print("==========================================\n")

# ==========================================
# 5. Train Final Model & Predict (FIXED)
# ==========================================
print("Rebuilding pipeline with the BEST discovered parameters...")

# Build the final XGBoost model directly from Optuna's best dictionary
final_xgb = XGBClassifier(
    n_estimators=study.best_trial.params['xgb_n_estimators'],
    max_depth=study.best_trial.params['xgb_max_depth'],
    learning_rate=study.best_trial.params['xgb_learning_rate'],
    subsample=study.best_trial.params['xgb_subsample'],
    colsample_bytree=study.best_trial.params['xgb_colsample_bytree'],
    reg_alpha=study.best_trial.params['xgb_reg_alpha'],
    reg_lambda=study.best_trial.params['xgb_reg_lambda'],
    eval_metric='auc',
    random_state=42,
    tree_method='hist', # Your Mac Speed optimization
    n_jobs=-1
)

# Put it inside the pipeline (Notice: no elastic_net_selector here anymore!)
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', final_xgb)
])

print("Fitting final XGBoost pipeline on the ENTIRE training dataset...")
final_pipeline.fit(X, y)

print("Generating predictions for submission...")
test_predictions = final_pipeline.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': test_predictions
})

submission_df.to_csv('target_encoded_xgboost_submission.csv', index=False)
print("Finished! Saved perfectly tuned predictions to 'target_encoded_xgboost_submission.csv'.")