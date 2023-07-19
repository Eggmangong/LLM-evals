import pandas as pd

# 读取两个excel文件
df_clustered_prompts = pd.read_excel('/Users/jinqigong/Desktop/1.xlsx')
df_results = pd.read_excel('/Users/jinqigong/Desktop/Research/OpenAI Evals/Comparison/0711.xlsx')

# 使用"数据集"列合并两个数据集
df_merged = pd.merge(df_clustered_prompts, df_results, on="Dataset", how='left')

# 将结果保存到新的excel文件
df_merged.to_excel('2.xlsx', index=False)
