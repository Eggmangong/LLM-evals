from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from svm_dt_all import data_loader
from joblib import dump
from sklearn.impute import SimpleImputer

llm_columns = ['chatgpt', 'v1_chatglm', 'v2_chatglm',
               'chatglm-6b', 'gpt4all-13b-snoozy', 'vicuna-33b-v1.3',
               'guanaco-33b-merged', 'chatglm2-6b', 'vicuna-7b-v1.3',
               'vicuna-13b-v1.1', 'guanaco-65b-merged', 'dolly-v2-12b',
               'vicuna-13b-v1.3', 'vicuna-7b-v1.1']

X_test_imputed, X_test, y_test, y_temp, X_temp_imputed = data_loader()

# Train and save a Random Forest model for each LLM
for llm in llm_columns:
    y_train = y_temp[llm].fillna(0)
    y_test_llm = y_test[llm].fillna(0)  # Create a new series for each LLM

    # Train Random Forest model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_temp_imputed, y_train)
    dump(rf, f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/RFs/rf_{llm}.joblib")

    # Train Gradient Boosting model
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_temp_imputed, y_train)
    dump(gb, f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/GBs/gb_{llm}.joblib")
