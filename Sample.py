import pandas as pd
from sklearn.model_selection import train_test_split


def sample_data(input_file, output_file, sample_ratio=0.8, random_seed=42):
    """
    从原始CSV文件中抽取指定比例的数据并保存到新文件。

    参数:
    - input_file: 原始数据文件路径。
    - output_file: 输出数据文件路径。
    - sample_ratio: 抽取的数据比例，默认为20%。
    - random_seed: 随机种子，确保结果可重复。
    """
    # 加载原始数据
    df = pd.read_csv(input_file)

    # 抽取指定比例的数据
    df_sample, _ = train_test_split(df, test_size=1 - sample_ratio, random_state=random_seed)

    # 保存抽取的数据到新文件
    df_sample.to_csv(output_file, index=False)

    print(f"Sampled data saved to {output_file}. Total samples: {len(df_sample)}.")


# 调用函数，设置输入文件、输出文件和抽样比例
input_file_path = '/data/Data/data/cxd/Graph-Mamba-main/dataset/raw/Ton.csv'  # 修改为您的数据文件路径
output_file_path = '/data/Data/data/cxd/Graph-Mamba-main/dataset/raw/Ton2.csv'  # 修改为您想要保存的输出文件路径

sample_data(input_file_path, output_file_path, sample_ratio=0.2)

# import pandas as pd
#
# # 加载CSV文件
# df = pd.read_csv('/data/Data/data/cxd/Graph-Mamba-main/dataset/raw/Final_NFBot.csv')
#
# # 计算label列中0和1的数量
# label_counts = df['label'].value_counts()
# #
# # print(label_counts)
#
# import pandas as pd
#
# # 步骤2: 读取数据
# df = pd.read_csv('/data/Data/data/cxd/Graph-Mamba-main/dataset/raw/Final_NFBOT_2.csv')
#
# # 步骤3: 随机打乱数据
# # frac=1意味着返回所有行，random_state可以设置为任何数字，以确保可重复性
# shuffled_df = df.sample(frac=1, random_state=42)
#
# # 步骤4: 保存到新的CSV文件
# shuffled_df.to_csv('/data/Data/data/cxd/Graph-Mamba-main/dataset/raw/Final_NFBOT_2.csv', index=False)

