import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from joblib import dump


def data_loader():
    # Load the data and embeddings
    data = pd.read_excel("/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/results_complete.xlsx")
    embeddings = np.load("/Users/jinqigong/Desktop/Research/OpenAI Evals/Cluster/embeddings.npy")

    # Add embeddings to dataframe
    embed_df = pd.DataFrame(embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])])
    data_extended = pd.concat([data, embed_df], axis=1)

    # Create a list of all LLMs
    llm_columns = ['chatgpt', 'v1_chatglm', 'v2_chatglm',
                'chatglm-6b', 'gpt4all-13b-snoozy', 'vicuna-33b-v1.3',
                'guanaco-33b-merged', 'chatglm2-6b', 'vicuna-7b-v1.3',
                'vicuna-13b-v1.1', 'guanaco-65b-merged', 'dolly-v2-12b',
                'vicuna-13b-v1.3', 'vicuna-7b-v1.1']

    # Separate features and the scores of LLMs
    X = data_extended.drop(columns=["Dataset", "original_prompt"] + llm_columns)

    # Split the data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, data_extended[llm_columns], test_size=0.1, random_state=42)

    # Impute missing values with 0
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_temp_imputed = imputer.fit_transform(X_temp)
    X_test_imputed = imputer.transform(X_test)

    # print(data.columns)

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

    return X_test_imputed, X_test, y_test