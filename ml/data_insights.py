import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

import datetime
import os

from func_utils import set_chinese_font

"""
    特征处理模块
    1. 相关性矩阵       ->    correlation_matrix
    2. 缺失值占比       ->    miss_value_ratio
    3. 特征洞察         ->    field_data_insights
"""


def correlation_matrix(df: pd.DataFrame, params: dict = {} ):
    """
     对df所有字符类型的列计算相关系数矩阵,只保留下矩阵, 并作图
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - save_path (str): 图片保存路径, 默认为'./pic_files'。

    Returns:
        相关系数矩阵图片
    """
    # 选取数值型
    df = df.select_dtypes(include=[np.number])
    # 计算相关系数矩阵
    corr = df.corr()

    # 只保留下三角矩阵
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 设置图形大小
    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热图
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # 添加标题和坐标轴标签
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 保存图片
    pic_save_path = params.get('save_path', "./pic_files")
    filename = f"correlation_matrix_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    full_path = os.path.join(pic_save_path, filename)
    plt.savefig(full_path, bbox_inches='tight')

    plt.show()


def miss_value_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算df中每个列的缺失值占比
    Args:
        df (pd.DataFrame): 原始数据

    Returns:
        pd.DataFrame: 每个列名及其缺失值占比%
    """
    # 检查输入数据是否是一个DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # 创建一个新的DataFrame来存储结果
    results = pd.DataFrame()

    # 计算每列的总数据量
    total_counts = df.shape[0]

    # 如果DataFrame为空，返回空的结果DataFrame
    if total_counts == 0:
        return results

    # 计算每个列的缺失值数量和占比
    missing_counts = df.isnull().sum()
    missing_ratio = (missing_counts / total_counts) * 100

    # 将结果存储在DataFrame中
    results['Column'] = missing_counts.index
    results['Missing Value Ratio (%)'] = missing_ratio.values

    return results


def field_data_insights(df: pd.DataFrame, params: dict):
    """
    统计df中指定字段的相关信息
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - field_name (str): 指定字段名
            - field_type (str): 指定字段类型, ['text', 'numeric']
            - save_path (str):  图片保存路径, 默认为'./pic_files'
    示例：
    params = {
            'field_name': 'age',
            'field_type': 'numeric',
            'save_path': './'
        }

    图片规则：
    1. 如果字段类型为文本型，展示该字段的词频柱状统计图
       同时返回一个json，字段包括：最长文本字符数，最短文本字符数
    2. 如果字段类型为数值型，展示该字段的直方图，核密度估计图，箱线图，小提琴图
       同时返回一个json，字段包括：最大值，最小值，均值，下四分位数，标准差，相关系数，中位数，上四分位数

    Returns:
        dict: 返回统计信息的json数据
    """
    set_chinese_font()  # 设置中文字体
    field_name = params['field_name']
    field_type = params['field_type']

    # 保存图片路径
    pic_save_path = params.get('save_path', "./pic_files")
    filename = f"field_data_insights_{field_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    full_path = os.path.join(pic_save_path, filename)

    if field_type == 'text':
        # 文本型数据处理
        df[field_name] = df[field_name].astype(str)
        text_lengths = df[field_name].apply(len)
        text_stats = {
            "最长文本字符数": text_lengths.max(),
            "最短文本字符数": text_lengths.min()
        }

        # 绘制词频柱状图
        plt.figure(figsize=(10, 6))
        df[field_name].value_counts().head(10).plot(kind='bar')
        plt.title('词频柱状统计图')
        plt.xlabel(field_name)
        plt.ylabel('频率')

        plt.savefig(full_path, bbox_inches='tight')
        plt.show()

        return text_stats

    elif field_type == 'numeric':
        # 数值型数据处理
        numeric_stats = {
            "最大值": df[field_name].max(),
            "最小值": df[field_name].min(),
            "均值": df[field_name].mean(),
            "下四分位数": df[field_name].quantile(0.25),
            "标准差": df[field_name].std(),
            "方差": df[field_name].var(),
            "中位数": df[field_name].median(),
            "上四分位数": df[field_name].quantile(0.75)
        }

        # 绘制图表
        plt.figure(figsize=(14, 10))

        # 直方图
        plt.subplot(2, 2, 1)
        df[field_name].plot(kind='hist', bins=20, color='blue', edgecolor='black')
        plt.title('直方图')

        # 核密度估计图
        plt.subplot(2, 2, 2)
        sns.kdeplot(df[field_name], fill=True)
        plt.title('核密度估计图')

        # 箱线图
        plt.subplot(2, 2, 3)
        sns.boxplot(df[field_name])
        plt.title('箱线图')

        # 小提琴图
        plt.subplot(2, 2, 4)
        sns.violinplot(df[field_name])
        plt.title('小提琴图')

        plt.tight_layout()

        plt.savefig(full_path, bbox_inches='tight')
        plt.show()

        return numeric_stats, full_path



if __name__ == '__main__':
    # 创建示例数据
    iris_datas = datasets.load_iris()
    # iris_df = pd.DataFrame(iris_datas.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    iris_df = pd.DataFrame(iris_datas.data, columns=iris_datas.feature_names)
    iris_df['label'] = iris_datas.target
    test_data = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')

    params = {
        'field_name': 'income',
        'field_type': 'numeric'
    }

    # 调用函数生成相关系数矩阵图
    # correlation_matrix(test_data)
    # print(miss_value_ratio(test_data))
    print(field_data_insights(test_data, params))
