import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Binarizer, OrdinalEncoder
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
import numpy as np

"""
    特征处理模块
    1. 选取X/Y列       ->    selected_label_df
    2. 特征编码        ->    feature_encoder
    3. 特征交叉        ->    feature_crossover
    4. 特征分箱        ->    feature_bin
    5. 特征选择        ->    feature_selection
"""


def selected_label_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    选取X/Y列
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - feature_cols (list): 特征列名列表
            - label_col (str):  标签列名

    示例：
    params = {
        'feature_cols': ['name', 'age', 'gender'],
        'label_col': 'label'
    }

    Returns:
        默认最后一列为标签列的df
    """
    # 验证参数中是否包含所需的键
    if 'feature_cols' not in params or 'label_col' not in params:
        raise ValueError("params 字典中必须包含 'feature_cols' 和 'label_col'。")

    # 检查是否所有列名都在 DataFrame 中
    feature_cols = params['feature_cols']
    label_col = params.get('label_col', 'label')
    required_cols = feature_cols + [label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"以下列名在 DataFrame 中缺失：{missing_cols}")

    # 根据 feature_cols 和 label_col 选取数据
    selected_df = df[feature_cols + [label_col]]

    return selected_df


def feature_encoder(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    从df中选取指定列进行特征编码,其他列不变
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - new_col_suffix (str): 新生成的字段后缀名称。
            - is_add_new_col: (bool) 是否生成新列
                                默认: True 新增编码后的列，False 输出编码后的列名为原列名

            - target_field (dic): 用于处理的字段名称和编码方法。
                - encode_method (str): 处理方法，'OneHotEncoder'，'LabelEncoder' 或'Binarizer'。
                - threshold (float): 用于数值型数据二值化的阈值

    示例：
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

    Returns:
        编码后的 DataFrame
    """

    # 参数解析与默认值设置
    new_col_suffix = params.get('new_col_suffix', '_encoded')
    is_add_new_col = params.get('is_add_new_col', True)

    # 逐一处理各字段
    for field, enc_info in params.items():
        if field not in ['new_col_suffix', 'is_add_new_col']:
            encode_method = enc_info.get('encode_method')
            if encode_method not in ['OneHotEncoder', 'LabelEncoder', 'Binarizer']:
                raise ValueError(f"Unsupported encoding method: {encode_method}")

            # 应用不同的编码方法
            if encode_method == 'OneHotEncoder':
                encoder = OneHotEncoder(sparse=False)
                transformed_data = encoder.fit_transform(df[[field]])
                cols = [field + f"_{c}" for c in encoder.categories_[0]]
                if is_add_new_col:
                    df[cols] = transformed_data
                else:
                    df.drop(columns=[field], inplace=True)
                    df[cols] = transformed_data

            elif encode_method == 'LabelEncoder':
                encoder = LabelEncoder()
                transformed_data = encoder.fit_transform(df[field])
                if is_add_new_col:
                    df[field + new_col_suffix] = transformed_data
                else:
                    df[field] = transformed_data

            elif encode_method == 'Binarizer':
                threshold = enc_info.get('threshold', 0)
                encoder = Binarizer(threshold=threshold)
                transformed_data = encoder.transform(df[[field]].values)
                if is_add_new_col:
                    df[field + new_col_suffix] = transformed_data
                else:
                    df[field] = transformed_data

    return df


def feature_crossover(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    选取df指定字段进行特征交叉
        遍历conf_lists，对feature_cols中的所有字段按照crossover_methods中的方法依次进行运算，生成新的字段
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            -conf_lists (list): 交叉字段配置JSON列表
                    {
                    -feature_cols (list): 交叉字段名列表
                    -crossover_methods (list): 交叉方法名列表， 如 ['add','subtract', 'multiply', 'divide']
                }

    示例：
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
    Returns:
        交叉后的 DataFrame
    """
    for conf in params['conf_lists']:
        feature_cols = conf['feature_cols']
        crossover_methods = conf['crossover_methods']

        # 对每种方法处理所有字段
        for method in crossover_methods:
            # 初始化结果列
            if method == 'add':
                result_col = df[feature_cols].sum(axis=1)
            elif method == 'subtract':
                result_col = df[feature_cols].diff(axis=1).iloc[:, -1]  # 计算最终的差异
            elif method == 'multiply':
                result_col = df[feature_cols].prod(axis=1)
            elif method == 'divide':
                # 避免除以0的错误
                result_col = df[feature_cols].replace(0, np.nan)
                for col in feature_cols[1:]:  # 除第一个列以外的所有列
                    result_col = result_col.div(df[col], axis=0)

            # 生成新列名并将结果列添加到DataFrame
            new_col_name = f"{'_'.join([f'{col}_{method}' for col in feature_cols])[:-len(method) - 1]}"
            df[new_col_name] = result_col

    return df


def feature_bin(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    对df中指定字段按照指定的分箱方法进行分箱处理
    然后对分箱后的结果进行位置编码
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典
            - bin_method (str): 分箱方法
                        包括等宽分箱（EqualWidth）、等频分箱（EqualFrequency）、
                        基于分位数的分箱（QuantileBased）、以及基于数据的平均值和标准差（MeanStdBased）
            - field_name (dic): 需要分箱的字段名
                - conf_value (): 分箱相关配置参数
                - colum_suffix (str): 分箱后新生成的字段后缀名称

    示例：
    params = {
              "bin_method": "EqualWidth",
              "age": {
                "conf_value": 5,
                "colum_suffix": "_binned"
              },
              "weight": {
                "conf_value": 30,
                "colum_suffix": "_binned"
              }
            }

    Returns:
        分箱后的 DataFrame
    """
    bin_method = params.get('bin_method')
    if bin_method not in['EqualWidth', 'EqualFrequency', 'QuantileBased', 'MeanStdBased']:
        raise ValueError(f"Unsupported bin method: {bin_method}")

    encoder = OrdinalEncoder()  # 创建OrdinalEncoder实例
    for field, conf in params.items():
        if field not in ['bin_method']:
            conf_value = conf.get('conf_value')
            colum_suffix = conf.get('colum_suffix', '_binned')
            new_field_name = field + colum_suffix

            # 根据分箱方法分箱
            if bin_method == 'EqualWidth':
                bin_width = conf_value
                bins = np.arange(0, df[field].max() + bin_width, bin_width)
                df[new_field_name] = pd.cut(df[field], bins=bins)
            elif bin_method == 'EqualFrequency':
                df[new_field_name] = pd.qcut(df[field], q=conf_value, duplicates='drop')
            elif bin_method == 'QuantileBased':
                quantiles = np.linspace(0, 1, conf_value + 1)
                df[new_field_name] = pd.qcut(df[field], q=quantiles, duplicates='drop')
            elif bin_method == 'MeanStdBased':
                mean = df[field].mean()
                std = df[field].std()
                conf_map = {
                    "1倍标准差": 1,
                    "2倍标准差": 2,
                    "3倍标准差": 3,
                }
                std_multiplier = conf_map.get(conf_value, 1)
                edges = [-np.inf, mean - std_multiplier * std, mean, mean + std_multiplier * std, np.inf]
                df[new_field_name] = pd.cut(df[field], bins=edges)

            # 将分箱结果进行Ordinal编码
            df[new_field_name] = encoder.fit_transform(df[[new_field_name]].astype(str))

    return df


def feature_selection(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    对df中指定字段，按照不同的字段类型跟标签列进行指定方法检验，从而进行特征选择
    选取p值小于0.05的特征列
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典
            - label_field (str): 标签列名
            - label_type (str): 标签列类型, [string, numerical]
            - feature_field_name (dict): 需要进行特征选择的字段名及其类型
                - field_type (str): 字段类型
                - p_value (float): 检验的p值，默认0.05

    示例：
    params = {
              "label_field": "label",
              "label_type": "string",
              "age": {
                "field_type": "numerical",
                "p_value": 0.05
                },
              "weight": {
                "field_type": "numerical",
                "p_value": 0.05
                }
            }

    检验规则：
        - 将label_field的字段分别跟参数里的其他字段进行相关性检验
        - 如果label_type和field字段里的field_type类型都是numerical，就用t检验
        - 如果label_type和field字段里的field_type类型都是string，就用卡方检验
        - 如果label_type和field字段里其中一个类型是string，另一个是numerical,就用F检验
        - 最后根据p值进行筛选，返回过滤后的DataFrame

    Returns:
        特征选择后的 DataFrame
    """
    # 确保标签字段和未经过特征选择的其他字段都被包括
    selected_features = [col for col in df.columns if col not in params or col == params['label_field']]
    label_field = params['label_field']
    label_type = params['label_type']

    for field, conf in params.items():
        if field in df.columns and field not in ['label_field', 'label_type'] and isinstance(conf, dict):
            field_type = conf.get('field_type')
            p_value_threshold = conf.get('p_value', 0.05)

            if label_type == 'numerical' and field_type == 'numerical':
                # Perform t-test
                stat, p_value = ttest_ind(df[label_field].dropna(), df[field].dropna(), nan_policy='omit')
            elif label_type == 'string' and field_type == 'string':
                # Prepare contingency table and perform chi-square test
                contingency_table = pd.crosstab(df[label_field], df[field])
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
            elif (label_type == 'string' and field_type == 'numerical') or (
                    label_type == 'numerical' and field_type == 'string'):
                # Prepare groups for F-test
                groups = [group.dropna() for name, group in df.groupby(label_field)[field]]
                stat, p_value = f_oneway(*groups)

            # Check if the field meets the p-value threshold

            if p_value < p_value_threshold:
                selected_features.append(field)

    return df[selected_features]




if __name__ == '__main__':
    # 示例使用和测试
    data = {
        'age': [3, 5, 2, 8, 3],
        'text': ['cat', 'dog', 'bird', 'dog', 'cat'],
        'name': ['Tom', 'Jerry', 'Alice', 'Bob', 'Tom']
    }
    df = pd.DataFrame(data)
    test_data = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')
    print(df)
    params = {
        'new_col_suffix': '_encoded',
        'is_add_new_col': True,
        'age': {
            'encode_method': 'Binarizer',
            'threshold': 0.5
        },
        'text': {
            'encode_method': 'LabelEncoder'
        }
    }

    encoded_df = feature_encoder(test_data, params)
    print(encoded_df)
