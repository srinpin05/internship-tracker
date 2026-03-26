import pandas as pd
from scipy.stats import rankdata

print("Initiating Final Rank Average Blend...")

# 1. Load your absolute best submission (The 0.9397 file)
best_sub = pd.read_csv('catboost_multiseed_stack_submission.csv')

# 2. Load your SECOND best submission (Change this filename!)
second_best_sub = pd.read_csv('catboost_kfold_final_push.csv')

# 3. Convert raw probabilities into Rankings (1st, 2nd, 3rd... Nth)
best_ranks = rankdata(best_sub['Y'])
second_ranks = rankdata(second_best_sub['Y'])

# 4. Blend the ranks (Giving 75% power to your champion, 25% to the backup)
final_blended_ranks = (best_ranks * 0.75) + (second_ranks * 0.25)

# 5. Normalize the ranks back into a 0.0 to 1.0 scale for the AUC metric
final_probs = final_blended_ranks / final_blended_ranks.max()

# 6. Save the final buzzer-beater submission
submission_df = pd.DataFrame({
    'Id': best_sub['Id'],
    'Y': final_probs
})

submission_df.to_csv('blend.csv', index=False)
print("Finished! Saved 'blend.csv'.")