"""
  function: test_sample
"""
import pandas as pd
from data_preprocess import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 设置显示选项，将浮点数的显示格式设置为完整数值
pd.set_option('display.float_format', lambda x: '%.0f' % x)

data = pd.read_csv("E:\\datasets\\titanic.csv")
test_data = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')


def test_balance_sample_by_random():
    # 随机欠采样

    params_absolute = {
        'target_field': 'Survived',  # 替换为实际的目标列名
        'random_type': 'absolute',  # 指定抽样类型为绝对数量
        'random_seed': 42,  # 随机种子，确保可重现性
        'sample_cls_params': {
            0: 5,  # 类别0的样本数量
            1: 3  # 类别1的样本数量
        }
    }
    balanced_data_down = balance_sample_by_random(data, params_absolute)
    print(balanced_data_down.groupby('Survived').size())


def test_balance_sample_by_smote():
    # 随机过采样

    params_smote = {
        'target_field': 'Survived',  # 替换为实际的目标列名
        'random_seed': 42,  # 随机种子，确保可重现性
        'k_neighbors': 5  # 用于计算 SMOTE 的近邻数
    }

    balanced_data_over = balance_sample_by_smote(data, params_smote)
    print(balanced_data_over.groupby('Survived').size())


def test_remove_duplicates_data():
    # 去除重复数据
    remove_params = {
        'target_fields': ['name', 'country'],
        'ignore_case': True
    }

    removed_data = remove_duplicates_data(test_data, remove_params)
    print(removed_data)


def test_fill_nan_data():
    # 填充缺失数据

    fill_params = {
        'age': {
            'field_type': '数值',
            'fill_method': '众数',  # 用平均年龄填充缺失的年龄
        },
        'name': {
            'field_type': '文本',
            'fill_method': '最多次数项'
        },
        'join_date': {
            'field_type': '日期',
            # 'fill_method': 'constant',
            'fill_value': '2024-01-01'  # 用指定日期填充缺失的加入日期
        }
    }

    filled_df = fill_nan_data(test_data, fill_params)
    print(filled_df)


def test_data_type_convert():
    # 类型转换
    params = {
        'age': {
            'field_type': '数值',
            'process_method': '数值转文本',
            'conf_params': {
                '区间转换': {
                    "0~18": "少年",
                    "19~30": "青年",
                    "31~50": "中年",
                    "51~100": "老年"
                }
            }
        },
        'label': {
            'field_type': '数值',
            'process_method': '数值转文本',
            'conf_params': {
                '唯一值转换': {
                    1: "a",
                    2: "b",
                    3: "c"
                }
            }
        },
        'join_date': {
            'field_type': '日期',
            'process_method': '日期转数值',
            'conf_params': {
                'date_format': '%Y-%m-%d'
            }
        },
        'join_date2': {
            'field_type': '日期',
            'process_method': '日期转文本',
            'conf_params': {
                '区间转换': {
                    "2024-06-01~2024-06-30": "五一节后",
                    "2024-05-01~2024-05-06": "五一节中"

                },
                'date_format': '%Y-%m-%d'
            }
        },
        'text': {
            'field_type': '文本',
            'process_method': '文本转数值',
            'conf_params': {
                'unique_value_list': [],
                'rule_map': {
                  "re": "re.search('test', x)",  # 如果text字段文本中包含test，则转换为1
                  "value": 1
                }
            }
        }
    }

    converted_df = data_type_convert(test_data, params)
    print(converted_df)