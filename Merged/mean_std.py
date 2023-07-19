import pandas as pd

# 读取合并的excel文件
df_merged = pd.read_excel('/Users/jinqigong/Desktop/merged_new_9kmeans.xlsx')

# 计算每个cluster的accuracy的标准差
df_mean = df_merged.groupby('cluster_y')['Accuracy'].mean()
df_std = df_merged.groupby('cluster_y')['Accuracy'].std()

# 输出每个cluster的accuracy的标准差
print(df_mean)
print(df_std)
print(df_std / df_mean)