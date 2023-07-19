# 文件名：cluster_embeddings_new.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 加载嵌入
embeddings = np.load('/Users/jinqigong/Desktop/Research/OpenAI Evals/Cluster/embeddings.npy')

# 计算不同聚类数量下的轮廓系数
silhouettes = []
range_n_clusters = range(2, 10)  # 假设你想测试的聚类数量范围是2到10
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    silhouette_avg = silhouette_score(embeddings, clusters)
    silhouettes.append(silhouette_avg)

# 找到轮廓系数最高的聚类数量
best_n_clusters = range_n_clusters[silhouettes.index(max(silhouettes))]

# 使用最佳的聚类数量进行聚类
kmeans = KMeans(n_clusters=best_n_clusters)
clusters = kmeans.fit_predict(embeddings)

# 读取原始Excel文件，添加聚类结果，并保存到新的Excel文件中
df = pd.read_excel('/Users/jinqigong/Desktop/Research/OpenAI Evals/new_prompt/Rewrite/original_prompt_complete.xlsx')
df['cluster'] = clusters
df.to_excel('clustered_prompts_new.xlsx')
