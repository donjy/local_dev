import pandas as pd
from imblearn.over_sampling import SMOTE
from ML.func_utils import *


def balance_sample_by_random(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    随机欠采样 平衡样本，可按绝对数量或百分比。

    参数:
        df (pd.DataFrame): 需要抽样的数据集。
        params (dict): 包含所有必要参数的字典，包括：
            - target_field (str): 用于抽样的目标字段。
            - random_type (str): 抽样类型，'absolute' 表示固定数量，'percent' 表示比例。
            - random_seed (float): 随机数生成器的种子，确保可重现性。
            - sample_cls_params (dict): 指定每个类别的样本数量或比例。
              示例：{'class0': 100, 'class1': 150} 或 {'class0': 0.5, 'class1': 0.3}

        示例:
        params = {
            'target_field': 'Survived',  # 替换为实际的目标列名
            'random_type': 'absolute',  # 指定抽样类型为绝对数量
            'random_seed': 42,  # 随机种子，确保可重现性
            'sample_cls_params': {
                0: 5,  # 类别0的样本数量
                1: 3  # 类别1的样本数量
            }
         }
    返回:
        pd.DataFrame: 根据指定抽样类型平衡后的数据集。
    """

    # 提取参数
    target_field = params['target_field']
    random_type = params['random_type']
    random_seed = params['random_seed']
    sample_cls_params = params['sample_cls_params']

    # 验证目标字段
    if target_field not in df.columns:
        raise ValueError(f"目标字段 '{target_field}' 不存在于数据集中")

    # 验证目标字段是否是二分类字段
    if len(df[target_field].unique()) != 2:
        raise ValueError(f"目标字段 '{target_field}' 必须是二分类字段")

    # 校验抽样类型
    if random_type not in ['absolute', 'percent']:
        raise ValueError("random_type 必须是 'absolute' 或 'percent'")

    balanced_df_list = []
    sampled_subset = None

    for cls, sample_param in sample_cls_params.items():
        subset = df[df[target_field] == cls]

        if random_type == 'absolute':
            sampled_subset = subset.sample(n=sample_param, random_state=random_seed)
        elif random_type == 'percent':
            sampled_subset = subset.sample(frac=sample_param, random_state=random_seed)

        balanced_df_list.append(sampled_subset)

    # 合并抽样数据并重置索引
    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)

    return balanced_df


def balance_sample_by_smote(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    使用 SMOTE 过采样算法平衡样本。
    通常要求数据集都是数值型的，因为 SMOTE 算法是基于距离计算的，这里会将非数值型数据转换为数值型进行处理。
    Args:
        df (pd.DataFrame): 需要抽样的数据集。
        params (dict): 包含所有必要参数的字典，包括：
            - target_field (str): 用于抽样的目标字段。
            - random_seed (float): 随机数生成器的种子，确保可重现性。
            - k_neighbors (int): 用于计算 SMOTE 的近邻数，默认为 5。
        示例：
        params = {
            'target_field': 'Survived',  # 替换为实际的目标列名
            'random_seed': 42,  # 随机种子，确保可重现性
            'k_neighbors': 5  # 用于计算 SMOTE 的近邻数
        }

    Returns:
        pd.DataFrame: 根据 SMOTE 算法平衡后的数据集。
    """
    # 提取参数
    target_field = params['target_field']
    random_seed = params['random_seed']
    k_neighbors = params.get('k_neighbors', 5)

    # 验证目标字段
    if target_field not in df.columns:
        raise ValueError(f"目标字段 '{target_field}' 不存在于数据集中")

    # 验证目标字段是否是二分类字段
    if len(df[target_field].unique()) != 2:
        raise ValueError(f"目标字段 '{target_field}' 必须是二分类字段")

    # 保留原始数据副本并对数据进行预处理，同时获取映射
    df_original = df.copy()
    df_preprocessed, mappings = preprocess_data(df, return_mappings=True)

    # 缺失值处理, 使用后向填充（bfill）
    df_preprocessed = df_preprocessed.bfill()
    # 仍然存在NaN的情况下用均值填充
    df_preprocessed = df_preprocessed.fillna(df.mean())

    # SMOTE
    smote = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
    X = df_preprocessed.drop(columns=[target_field])
    y = df_preprocessed[target_field]
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 将结果合并回DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_field] = y_resampled

    # 转换生成的数据回字符类型
    df_resampled = invert_mappings(df_resampled, mappings)

    # 标记数据来源：'original' 或 'smote'
    df_resampled['origin'] = 'smote'
    df_original['origin'] = 'original'

    # 合并原始数据和SMOTE生成的数据
    combined_df = pd.concat([df_original, df_resampled])
    combined_df.reset_index(drop=True, inplace=True)

    # 删除合并后的origin字段和索引
    combined_df.drop(columns=['origin'], inplace=True)

    return combined_df


def remove_duplicates_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    删除数据集中的重复数据。
    默认按所有列进行重复数据检查，如果需要指定列进行检查，请在参数中指定。
    Args:
        df: (pd.DataFrame): 需要抽样的数据集。
        params: (dict): 包含所有必要参数的字典，包括：
            - target_fields (list): 用于检查重复数据的字段列表，默认为所有字段。
            - ignore_case (bool): 是否忽略大小写，默认为 False。

        示例：
        params = {
            'target_fields': ['Name', 'country']  # 用于重复数据检查的字段列表
            'ignore_case': False  # 是否忽略大小写
    Returns:
        pd.DataFrame: 删除重复数据后的数据集。
    """
    # 提取参数
    duplicates_subset = params.get('target_fields', df.columns.tolist())  # 默认为所有列
    ignore_case = params.get('ignore_case', False)  # 默认大小写敏感

    # 检查指定的字段是否都存在于DataFrame中
    if not all(col in df.columns for col in duplicates_subset):
        missing_cols = [col for col in duplicates_subset if col not in df.columns]
        raise ValueError(f"以下指定的字段不存在于DataFrame中: {missing_cols}")

    # 如果需要忽略大小写
    if ignore_case:
        for col in duplicates_subset:
            if df[col].dtype == object:
                df[col] = df[col].str.lower()  # 转换为小写以忽略大小写差异

    # 删除重复数据
    df = df.drop_duplicates(subset=duplicates_subset, keep='first', inplace=False).reset_index(drop=True)

    return df


def fill_nan_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    使用指定的规则填充数据集中的缺失值。
    Args:
        df (pd.DataFrame): 需要填充缺失值的数据集。
        params (dict): 包含所有必要参数的字典，包括：
            - target_field (dic): 用于填充缺失值的字段名称以及填充方法。
                - field_type (str): 字段类型，'数值'，'文本' ，'日期'。
                - fill_method (str): 填充缺失值的方法：
                                     1. 针对数值型提供最小值、最大值、平均值、中位数和自定义等方式的缺失值处理
                                     2. 针对字符型及文本型提供最多次项、最少次项和自定义等方式的缺失值处理。
                                     3. 针对日期型提供自定义方式的缺失值处理。支持用户自定义设置一个特定的日期值赋给缺失的单元格
                - fill_value (str): 自定义填充值，当 fill_method 为 '自定义' 时使用。
        示例：
        params = {
            'age': {
                'field_type': '数值',
                'fill_method': 'mean'
            },
            'name': {
                'field_type': '文本',
                'fill_method': 'highest_frequency'
            }
        }

    Returns:
        pd.DataFrame: 填充缺失值后的数据集。
    """

    for field, specs in params.items():
        field_type = specs['field_type']
        fill_method = specs.get('fill_method', None)  # 获取填充方法，默认为None
        fill_value = specs.get('fill_value', None)  # 获取自定义填充值，默认为None

        if field not in df.columns:
            raise ValueError(f"字段 '{field}' 不存在于数据集中")

        if field_type == '数值':
            # 对数值型字段应用不同的填充策略
            if fill_method == '平均值':
                df[field].fillna(df[field].mean(), inplace=True)
            elif fill_method == '中位数':
                df[field].fillna(df[field].median(), inplace=True)
            elif fill_method == '众数':
                df[field].fillna(df[field].mode().iloc[0], inplace=True)
            elif fill_method == '最大值':
                df[field].fillna(df[field].max(), inplace=True)
            elif fill_method == '最小值':
                df[field].fillna(df[field].min(), inplace=True)
            else:
                if fill_value is not None:
                    df[field].fillna(fill_value, inplace=True)

        elif field_type == '文本':
            # 对文本型字段应用不同的填充策略
            if fill_method == '最多次数项':
                most_common = df[field].value_counts().idxmax()
                df[field].fillna(most_common, inplace=True)
            elif fill_method == '最少次数项':
                least_common = df[field].value_counts().idxmin()
                df[field].fillna(least_common, inplace=True)
            else:
                if fill_value is not None:
                    df[field].fillna(fill_value, inplace=True)

        elif field_type == '日期':
            # 日期字段只支持自定义填充
            if fill_value is not None:
                try:
                    # 先按日期格式填充，不行就按成字符类型填充
                    df[field].fillna(pd.to_datetime(fill_value), inplace=True)
                except ValueError:
                    df[field].fillna(fill_value, inplace=True)

    return df


def data_type_convert(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    将数据集中的字段转换为指定的数据类型。
    Args:
        df (pd.DataFrame): 需要填充缺失值的数据集。
        params (dict): 包含所有必要参数的字典，包括：
            - target_field (dic): 用于填充缺失值的字段名称以及填充方法。
            - field_type (str): 字段类型，'数值'，'文本' ，'日期'。
            - process_method: 处理方法，'数字转文本'，'数字转日期'，'日期转文本'， '日期转数字'， '文本转数字'，'文本转日期'
            - conf_params: 配置参数，如'区间转换'，'唯一值转换'
            示例：
            conf_params = {
                      "区间转换": {
                        "0~18": "少年",
                        "19~30": "青年",
                        "31~50": "中年",
                        "51~100": "老年"
                      },
                      "唯一值转换": {
                        "0": "少年",
                        "1": "老年"
                      }
                    }

    Returns: pd.DataFrame: 转换数据类型后的数据集。

    """
    
    convertor = TypeConversion(df)
    for field, specs in params.items():
        field_type = specs['field_type']
        process_method = specs['process_method']
        conf_params = specs['conf_params']

        # 数值转文本
        if field_type == '数值' and process_method == '数值转文本':
            if '区间转换' in conf_params:
                # 从配置中提取区间和标签，配对并排序
                bin_label_pairs = [
                    (int(k.split('~')[0]), int(k.split('~')[1]), v)
                    for k, v in conf_params['区间转换'].items()
                ]

                # 根据区间起始值排序配对
                bin_label_pairs.sort(key=lambda x: x[0])

                # 解包为排序后的bins和labels
                bins = [pair[0] for pair in bin_label_pairs] + [bin_label_pairs[-1][1]]
                labels = [pair[2] for pair in bin_label_pairs]
                convertor.numbers_convert_to_text_bins(field, bins, labels)
            elif '唯一值转换' in conf_params:
                convertor.numbers_convert_to_text_unique(field, conf_params['唯一值转换'])

        # 数值转日期
        elif field_type == '数值' and process_method == '数值转日期':
            date_format = conf_params.get('date_format', '%Y-%m-%d')
            convertor.numbers_convert_to_date(field, date_format)

        # 日期转文本
        elif field_type == '日期' and process_method == '日期转文本':
            date_format = conf_params.get('date_format', '%Y-%m-%d')

            # 从配置中提取日期区间和标签，转换为datetime对象并配对
            bin_label_pairs = [
                (pd.to_datetime(k.split('~')[0], format=date_format),
                 pd.to_datetime(k.split('~')[1], format=date_format), v)
                for k, v in conf_params['区间转换'].items()
            ]

            # 根据区间起始日期排序配对，并解包为排序后的bins和labels
            bin_label_pairs.sort(key=lambda x: x[0])  # 根据起始日期排序
            bins = [pair[0] for pair in bin_label_pairs] + [bin_label_pairs[-1][1]]
            labels = [pair[2] for pair in bin_label_pairs]

            convertor.date_convert_to_text_bins(field, bins, labels, date_format)

        # 日期转数值
        elif field_type == '日期' and process_method == '日期转数值':
            timestamp_length = conf_params.get('timestamp_length', 13)
            date_format = conf_params.get('date_format', '%Y-%m-%d')
            convertor.date_convert_to_timestamp(field, timestamp_length, date_format)

        # 文本转数值
        elif field_type == '文本' and process_method == '文本转数值':
            rule_map = conf_params.get('rule_map')
            convertor.text_convert_to_numbers(field, rule_map)

        # 文本转日期
        elif field_type == '文本' and process_method == '文本转日期':
            rule_map = conf_params.get('rule_map')
            convertor.text_convert_to_dates(field, rule_map)

    return convertor.df
