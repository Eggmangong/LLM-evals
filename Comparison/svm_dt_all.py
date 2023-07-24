from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
from data_loader import data_loader1

llm_columns = ['chatgpt', 'v1_chatglm', 'v2_chatglm',
               'chatglm-6b', 'gpt4all-13b-snoozy', 'vicuna-33b-v1.3',
               'guanaco-33b-merged', 'chatglm2-6b', 'vicuna-7b-v1.3',
               'vicuna-13b-v1.1', 'guanaco-65b-merged', 'dolly-v2-12b',
               'vicuna-13b-v1.3', 'vicuna-7b-v1.1']

X_test_imputed, X_test, y_test, y_temp, X_temp_imputed = data_loader1()

# Train and save a model for each LLM
for llm in llm_columns:
    y_train = y_temp[llm].fillna(0)
    y_test_llm = y_test[llm].fillna(0)  # Create a new series for each LLM

    # Train SVM model
    svm = SVR()
    svm.fit(X_temp_imputed, y_train)
    dump(svm, f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/SVMs/svm_{llm}.joblib")

    # Train Decision Tree model
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_temp_imputed, y_train)
    dump(tree, f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/DTs/tree_{llm}.joblib")





