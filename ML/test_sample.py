"""
  function: test_sample
"""
import pandas as pd
from data_preprocess import *
from feature_process import *
from model import *

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
        'age2': {
            'field_type': '数值',
            'process_method': '数值转文本',
            'conf_params': {
                '唯一值转换': {
                    'unique_value_list': [[2, 22], [50, 70]],
                    'value': ['年轻人', '老人']
                }
            }
        },
        'time': {
            'field_type': '数值',
            'process_method': '数值转日期',
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
        'date': {
            'field_type': '日期',
            'process_method': '日期转文本',
            'conf_params': {
                '唯一值转换': {
                    'unique_value_list': [['2021-05-01', '2023-12-31'], ['2024-01-01', '2024-1-1']],
                    'value': ['2021年', '2024年']

                },
                'date_format': '%Y-%m-%d'
            }
        },
        'join_date': {
            'field_type': '日期',
            'process_method': '日期转数值',
            'conf_params': {
                'date_format': '%Y-%m-%d'
            }
        },
        'text2': {
            'field_type': '文本',
            'process_method': '文本转数值',
            'conf_params': {
                'rule_map': {
                    'unique_value_list': [['123', '456', '789'], ['11', '22']],  # 指定文本字段的唯一值列表
                    're': [r"^\d{3}$", r"^\d{2}$"],
                    'value': [2, 3]
                }
            }
        },
        'text': {
            'field_type': '文本',
            'process_method': '文本转日期',
            'conf_params': {
                'rule_map': {
                    'unique_value_list': [['10/10/2020aaa'], ['2023-12-31', '2024-01-01', 'He arrived on 2023-06-15']],
                    "re": [r"\d{2}/\d{2}/\d{4}", r"\d{4}-\d{2}-\d{2}"],
                    'date_format': ['%m/%d/%Y', '%Y-%m-%d']
                }
            }
        }
    }

    converted_df = data_type_convert(test_data, params)
    print(converted_df)


def test_data_standard_normalise():
    # 数据标准化

    params = {
        'new_col_suffix': '_normalised',
        'output_all_cols': False,
        'age': {
            'normal_method': '最大最小归一化',
        },
        'age2': {
            'normal_method': 'Z标准化',
        }
    }

    standardised_df = data_standard_normalise(test_data, params)
    print(standardised_df)


def test_data_outlier_detect():
    # 异常值检测

    params = {
        'target_fields': {
            'age': {
                'detect_method': '基于四分位距',
                'judge_condition': '大于',
                'threshold': 1.5,
                'replace_value': 666,
            },
            'income': {
                'detect_method': '自定义异常检测公式',
                'judge_condition': '大于',
                'threshold': 200,
                'replace_value': 9999,
            }
        },
        'process_strategy': {
            'all_condition': True,
            'process_method': '直接删除',
        }
    }

    outlier_df = data_outlier_detect(test_data, params)
    print(outlier_df[['age', 'income']])


def test_selected_label_df():
    params = {
        'feature_cols': ['name', 'age', 'sex'],
        'label_col': 'label'
    }

    selected_df = selected_label_df(test_data, params)
    print(selected_df)


def test_feature_encoder():
    params = {
        'new_col_suffix': '_encoded',
        'is_add_new_col': True,
        'age': {
            'encode_method': 'Binarizer',
            'threshold': '2'
        },
        'text': {
            'encode_method': 'LabelEncoder'
        }
    }

    encoded_df = feature_encoder(test_data, params)
    print(encoded_df)


def test_feature_crossover():
    params = {
        "conf_lists": [
            {
                "feature_cols": [
                    "age",
                    "income",
                    "height"
                ],
                "crossover_methods": [
                    "multiply",
                    "add"
                ]
            },
            {
                "feature_cols": [
                    "height",
                    "weight"
                ],
                "crossover_methods": [
                    "multiply",
                    "add"
                ]
            }
        ]
    }

    crossover_df = feature_crossover(test_data, params)
    print(crossover_df)


def test_feature_bin():
    params = {
        "bin_method": "MeanStdBased",
        "age": {
            "conf_value": "2倍标准差",
            "colum_suffix": "_binned"
        },
        "weight": {
            "conf_value": "3倍标准差",
            "colum_suffix": "_binned"
        }
    }

    binned_df = feature_bin(test_data, params)
    print(binned_df)


def test_feature_selection():
    params = {
        "label_field": "label",
        "label_type": "string",
        "country": {
            "field_type": "string",
            "p_value": 0.05
        },
        "weight": {
            "field_type": "numerical",
            "p_value": 0.05
        }
    }

    selected_df = feature_selection(test_data, params)
    print(selected_df)


def test_lr_model():
    params = {
        'split_ratio': 80,
        'penalty_value': 0.01,
        'max_iter': 100,
        'tolerance': 0.01,
        'penalty_type': 'l2',
    }

    iris_datas = datasets.load_iris()
    iris_df = pd.DataFrame(iris_datas.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    iris_df['label'] = iris_datas.target

    lr_model = logistic_regression(iris_df, params)
    print(lr_model)