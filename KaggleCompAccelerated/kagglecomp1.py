import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from catboost import CatBoostClassifier
#Load Data
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

#Preprocess y and X data
y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

#Identify categorical data for Catboost

numeric_features = X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X.select_dtypes(exclude=['float64']).columns.tolist()

# Convert categoricals to strings (filling NaNs with a specific 'Missing' tag)
for col in categorical_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

#print(f"Detected {len(categorical_features)} categoricals and {len(numeric_features)} continuous floats.")

# define the optuna objective for Catboost
def objective(trial):
    cb_params = {
        'iterations': trial.suggest_int('iterations', 500, 1200),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 128, 255),
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }
    
    cb_model = CatBoostClassifier(**cb_params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    #loop to execute cross validation
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr,X_val = X.iloc[train_idx],X.iloc[val_idx]
        y_tr,y_val = y.iloc[train_idx],y.iloc[val_idx]
        
        cb_model.fit(
            X_tr, y_tr,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        aucs.append(cb_model.get_best_score()['validation']['AUC'])
    
    return np.mean(aucs)

#Execute Optuna Optimization

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best trial AUC: {study.best_trial.value:.5f}")
print("Best hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")


#train final model and predict
final_cb = CatBoostClassifier(
    **study.best_trial.params,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    thread_count=-1
)

#fit model on entire trainingdataset
final_cb.fit(X, y, cat_features=categorical_features)

print("Generating predictions for submission...")
test_predictions = final_cb.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Y': test_predictions
})

submission_df.to_csv('catboost_submission.csv', index=False)
print("Finished! Saved perfectly tuned predictions to 'catboost_submission.csv'.")