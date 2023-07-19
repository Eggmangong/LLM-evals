import openai
import pandas as pd

# 设置你的OpenAI密钥
openai.api_key = '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'

# 读取Excel文件
df = pd.read_excel('/Users/jinqigong/Desktop/prompt_rewrite3.xlsx')

# 初始化新的prompt列
df['new_prompt'] = ''

# 循环处理每一行
for index, row in df.iterrows():
    # 如果original_prompt为空，则跳过
    if pd.isnull(row['original_prompt']):
        continue
    
    # 添加指示性语句
    task_description = "Rewrite the following prompt to provide more detailed guidance to help the large language model to understand, and you can add some examples without changing their original meaning: "
    prompt_to_rewrite = task_description + row['original_prompt']

    # 使用ChatModels和ChatCompletion API
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", # 使用最新的大型对话模型，至2021年9月是gpt-3.5-turbo
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt_to_rewrite}
      ]
    )
    
    # 存储改写的结果
    df.at[index, 'new_prompt'] = response['choices'][0]['message']['content'].strip()

# 把新的DataFrame写回到Excel文件中
df.to_excel('/Users/jinqigong/Desktop/prompt_rewrite3.xlsx', index=False)
