from joblib import load
from itertools import combinations
import numpy as np
from svm_dt_all import data_loader

llm_columns = ['chatgpt', 'v1_chatglm', 'v2_chatglm',
               'chatglm-6b', 'gpt4all-13b-snoozy', 'vicuna-33b-v1.3',
               'guanaco-33b-merged', 'chatglm2-6b', 'vicuna-7b-v1.3',
               'vicuna-13b-v1.1', 'guanaco-65b-merged', 'dolly-v2-12b',
               'vicuna-13b-v1.3', 'vicuna-7b-v1.1']

X_test_imputed, X_test, y_test, y_temp, X_temp_imputed = data_loader()
# print("data loader over")

# Load the models
svms = {llm: load(f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/SVMs/svm_{llm}.joblib") for llm in llm_columns}
trees = {llm: load(f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/DTs/tree_{llm}.joblib") for llm in llm_columns}
rfs = {llm: load(f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/RFs/rf_{llm}.joblib") for llm in llm_columns}
gbs = {llm: load(f"/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/GBs/gb_{llm}.joblib") for llm in llm_columns}


# Make predictions on the test set
svm_predictions = {llm: model.predict(X_test_imputed) for llm, model in svms.items()}
tree_predictions = {llm: model.predict(X_test_imputed) for llm, model in trees.items()}
rf_predictions = {llm: model.predict(X_test_imputed) for llm, model in rfs.items()}
gb_predictions = {llm: model.predict(X_test_imputed) for llm, model in gbs.items()}

# Compare the predictions of each pair of models for each test sample
correct_counts_svm = 0
correct_counts_tree = 0
correct_counts_rf = 0
correct_counts_gb = 0
for i in range(X_test.shape[0]):
    for llm1, llm2 in combinations(llm_columns, 2):
        if (svm_predictions[llm1][i] > svm_predictions[llm2][i]) == (y_test[llm1].iloc[i] > y_test[llm2].iloc[i]):
            correct_counts_svm += 1
        if (tree_predictions[llm1][i] > tree_predictions[llm2][i]) == (y_test[llm1].iloc[i] > y_test[llm2].iloc[i]):
            correct_counts_tree += 1
        if (rf_predictions[llm1][i] > rf_predictions[llm2][i]) == (y_test[llm1].iloc[i] > y_test[llm2].iloc[i]):
            correct_counts_rf += 1
        if (gb_predictions[llm1][i] > gb_predictions[llm2][i]) == (y_test[llm1].iloc[i] > y_test[llm2].iloc[i]):
            correct_counts_gb += 1

# Calculate the accuracy
total_counts = len(llm_columns) * (len(llm_columns) - 1) / 2 * X_test.shape[0]
accuracy_svm = correct_counts_svm / total_counts
accuracy_tree = correct_counts_tree / total_counts
accuracy_rf = correct_counts_rf / total_counts
accuracy_gb = correct_counts_gb / total_counts


print("SVM accuracy:", accuracy_svm)
print("Decision Tree accuracy:", accuracy_tree)
print("Random Forest accuracy:", accuracy_rf)
print("Gradient Boosting accuracy:", accuracy_gb)
