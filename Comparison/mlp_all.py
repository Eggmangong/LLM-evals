from sklearn.neural_network import MLPRegressor
from data_loader import data_loader1
from joblib import dump

llm_columns = ['chatgpt', 'v1_chatglm', 'v2_chatglm',
               'chatglm-6b', 'gpt4all-13b-snoozy', 'vicuna-33b-v1.3',
               'guanaco-33b-merged', 'chatglm2-6b', 'vicuna-7b-v1.3',
               'vicuna-13b-v1.1', 'guanaco-65b-merged', 'dolly-v2-12b',
               'vicuna-13b-v1.3', 'vicuna-7b-v1.1']

X_test_imputed, X_test, y_test, y_temp, X_temp_imputed = data_loader1()

# Train and save a MLP model for each LLM
for llm in llm_columns:
    y_train = y_temp[llm].fillna(0)
    y_test_llm = y_test[llm].fillna(0)  # Create a new series for each LLM

    # Train MLP model
    mlp = MLPRegressor(random_state=42, max_iter=500, learning_rate='constant', learning_rate_init=0.1, activation='tanh')
    mlp.fit(X_temp_imputed, y_train)
    dump(mlp, f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/MLPs/mlp_{llm}.joblib")