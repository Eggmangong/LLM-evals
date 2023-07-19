import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_excel("/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/results_complete.xlsx")

# Load the embeddings
embeddings = np.load("/Users/jinqigong/Desktop/Research/OpenAI Evals/Cluster/embeddings.npy")

# Add embeddings to dataframe
embed_df = pd.DataFrame(embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])])
data_extended = pd.concat([data, embed_df], axis=1)

# Separate features and target
X = data_extended.drop(columns=["Dataset", "original_prompt", "chatgpt"])
y = data_extended["chatgpt"]

# Split the data into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

# Impute missing values with 0
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Impute missing values in target with 0
y_train_imputed = y_train.fillna(0)
y_val_imputed = y_val.fillna(0)

# Train SVM model
svm = SVR()
svm.fit(X_train_imputed, y_train_imputed)
svm_val_predictions = svm.predict(X_val_imputed)
svm_val_mse = mean_squared_error(y_val_imputed, svm_val_predictions)

# Train Decision Tree model
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train_imputed, y_train_imputed)
tree_val_predictions = tree.predict(X_val_imputed)
tree_val_mse = mean_squared_error(y_val_imputed, tree_val_predictions)

print(f"SVM Validation MSE: {svm_val_mse}")
print(f"Decision Tree Validation MSE: {tree_val_mse}")
