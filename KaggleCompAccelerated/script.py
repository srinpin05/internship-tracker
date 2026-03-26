import pandas as pd
# Load your final submission file
check = pd.read_csv('catboost_submission.csv')
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('testing_data.csv')

# Replace negative placeholders with NaNs 
#missing_placeholders = [-1, -2, -3, -4]
#train_df.replace(missing_placeholders, np.nan, inplace=True)
#test_df.replace(missing_placeholders, np.nan, inplace=True)

y = train_df['Y']
X = train_df.drop(columns=['Id', 'Y'])
test_ids = test_df['Id']
X_test = test_df.drop(columns=['Id'])

print(f"Total Rows: {len(check)}") # Should match testing_data.csv row count
print(f"Any Nulls?: {check['Y'].isnull().sum()}") # Should be 0
print(f"Top 5 Values:\n{check['Y'].head()}") # Should be decimals, not 0s or 1s
print(f"Average Prediction Score: {check['Y'].mean():.4f}")
print(f"Percentage of Training Y=1: {y.mean():.4f}")