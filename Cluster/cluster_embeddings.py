# 文件名：cluster_embeddings.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# 加载嵌入
embeddings = np.load('embeddings.npy')

# 计算相似度
similarity_matrix = cosine_similarity(embeddings)

# 聚类
kmeans = KMeans(n_clusters=5, n_init=10)  # 假设你想要分成5个类别
clusters = kmeans.fit_predict(similarity_matrix)

# 读取原始Excel文件，添加聚类结果，并保存到新的Excel文件中
df = pd.read_excel('/Users/jinqigong/Desktop/Research/OpenAI Evals/new_prompt/original_prompt_complete.xlsx')
df['cluster'] = clusters
df.to_excel('clustered_prompts.xlsx')
