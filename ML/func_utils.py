import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def preprocess_data(df, return_mappings=False):
    """
    将非数值型数据转换为数值型，并可选择返回映射。
    """

    mappings = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mappings[col] = {index: label for index, label in enumerate(le.classes_)}
    if return_mappings:
        return df, mappings
    return df


def invert_mappings(df, mappings):
    """
    使用保存的映射将数值型字段转换回字符类型。
    """
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df


def convert_date(text, regex, date_format):
    """
    用正则表达式从文本中提取日期，并尝试按照指定格式转换日期。
    如果匹配成功，转换日期格式；如果失败，返回原始文本。
    """
    match = re.search(regex, text)
    if match:
        try:
            date = datetime.strptime(match.group(), '%Y-%m-%d')  # 假设匹配到的日期格式是YYYY-MM-DD
            return date.strftime(date_format)
        except ValueError:
            return text  # 如果日期格式不正确或无法转换
    return text  # 如果没有找到匹配项，返回原文本


class TypeConversion:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    # 将指定字段的数字类型按指定区间转文本类型
    def numbers_convert_to_text_bins(self, field: str, bins: list, labels: list) -> pd.DataFrame:
        """
        将指定字段的数字根据给定的区间转换为文本标签。
        没在bin区间的值将保持原值。

        Args:
            field (str): 需要转换的字段名称。
            bins (list): 定义区间的边界列表。
            labels (list): 对应于每个区间的标签列表。

        Returns:
            pd.DataFrame: 转换后的DataFrame，其中未在任何区间的值保持不变。
        """
        # 使用 pd.cut 进行区间划分
        cut_series = pd.cut(self.df[field], bins=bins, labels=labels, right=False)

        # 转换 cut_series 为字符串类型，以便使用 fillna
        self.df[field] = cut_series.astype(str).fillna(self.df[field])

        return self.df

    # 将指定字段的数字类型按指定唯一值转文本类型
    def numbers_convert_to_text_unique(self, field: str, mapping: dict) -> pd.DataFrame:
        """
        将指定字段的数字转换为文本，根据传入的映射字典进行转换。
        未在映射中的值将保持原值。

        Args:
            field (str): 需要转换的字段名称。
            mapping (dict): 包含数字到文本的映射。

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """

        self.df[field] = self.df[field].map(mapping).fillna(self.df[field])
        return self.df

    def numbers_convert_to_date(self, field: str, date_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """
             将指定字段的数字类型转为日期格式，支持10位和13位的Unix时间戳。
             默认日期格式为'%Y-%m-%d'。仅转换10位（秒级）和13位（毫秒级）的时间戳，
             如果字段值不满足条件则不进行转换，保留原值。

             Args:
                 field (str): 需要转换的字段名称。
                 date_format (str): 日期格式，默认为'%Y-%m-%d'。

             Returns:
                 pd.DataFrame: 转换后的DataFrame，如果字段不存在或无可转换数据，则返回原DataFrame。
             """
        if field not in self.df.columns:
            print(f"字段 '{field}' 不存在于DataFrame中")
            return self.df

        # 创建一个新列来临时存放转换后的日期，避免直接影响原始列
        self.df['_tmp_date'] = pd.to_numeric(self.df[field], errors='coerce')

        # 确定哪些行是10位或13位的时间戳
        conditions = self.df['_tmp_date'].notna()
        timestamp_lengths = self.df.loc[conditions, '_tmp_date'].astype(str).str.len()
        conditions &= timestamp_lengths.isin([10, 13])

        # 转换10位和13位时间戳
        self.df.loc[conditions & (timestamp_lengths == 10), '_tmp_date'] = pd.to_datetime(
            self.df.loc[conditions & (timestamp_lengths == 10), '_tmp_date'], unit='s'
        ).dt.strftime(date_format)
        self.df.loc[conditions & (timestamp_lengths == 13), '_tmp_date'] = pd.to_datetime(
            self.df.loc[conditions & (timestamp_lengths == 13), '_tmp_date'], unit='ms'
        ).dt.strftime(date_format)

        # 将转换好的日期值写回原字段，未转换的保留原值
        self.df[field] = self.df['_tmp_date'].fillna(self.df[field])
        self.df.drop(columns=['_tmp_date'], inplace=True)  # 清理临时列

        return self.df

    def date_convert_to_text_bins(self, field: str, bins: list, labels: list,
                                  date_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """
        将指定的日期字段根据给定的时间范围划分，将时间范围转换为文本标签。
        时间格式默认为'%Y-%m-%d'。

        Args:
            field (str): 需要转换的日期字段。
            bins (list): 时间范围的边界，应为可解析为日期的字符串列表。
            labels (list): 对应于每个时间范围的标签。
            date_format (str): 用于解析和格式化日期的字符串，默认为'%Y-%m-%d'。

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """
        # 尝试将字段转换为日期格式，错误则转为NaT
        self.df[field] = pd.to_datetime(self.df[field], format=date_format, errors='coerce')

        # 确保bins是日期类型
        bins = pd.to_datetime(bins, format=date_format)

        # 使用pd.cut来根据bins和labels转换日期为分类文本
        self.df[field] = pd.cut(self.df[field], bins=bins, labels=labels, right=False)

        return self.df

    def date_convert_to_timestamp(self, field: str, timestamp_length: int = 13,
                                  date_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """
        将日期格式的指定字段转换为10位或13位的时间戳。
        默认日期格式为'%Y-%m-%d'，默认转换为13位的时间戳。

        Args:
            field (str): 需要转换的日期字段。
            timestamp_length (int): 时间戳的长度，支持10位（秒级）或13位（毫秒级）。
            date_format (str): 用于解析日期的格式，默认为'%Y-%m-%d'。

        Returns:
            pd.DataFrame: 转换后的DataFrame。如果字段值不满足日期格式条件则不进行转换，保留原值。
        """
        if field in self.df.columns:
            # 解析日期并本地化时间
            self.df['_tmp_timestamp'] = pd.to_datetime(self.df[field], format=date_format, errors='coerce').dt.tz_localize('Asia/Shanghai', ambiguous='raise')

            # 将本地时间转换为UTC时间戳
            if timestamp_length == 13:
                self.df['_tmp_timestamp'] = self.df['_tmp_timestamp'].dt.tz_convert('UTC').astype('int64') // 10 ** 6
            elif timestamp_length == 10:
                self.df['_tmp_timestamp'] = self.df['_tmp_timestamp'].dt.tz_convert('UTC').astype('int64') // 10 ** 9

            # 将转换好的时间戳值写回原字段，未转换的保留原值
            self.df[field] = self.df['_tmp_timestamp'].fillna(self.df[field])
            self.df.drop(columns=['_tmp_timestamp'], inplace=True)  # 清理临时列
        else:
            print(f"字段 '{field}' 不存在于DataFrame中")

        return self.df

    def text_convert_to_numbers(self, field: str, rule_map: dict) -> pd.DataFrame:
        """
        将指定字段的文本内容按照正则表达式转换为数字。

        Args:
            field (str): 需要转换的字段名称。
            rule_map (dict): 包含正则表达式和对应的数字。
                示例: {'re': "re.search('test', x) is None", 'value': 1}

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """
        if field not in self.df.columns:
            print(f"字段 '{field}' 不存在于DataFrame中")
            return self.df

        # 初始化列，假设未匹配到则保留原始值
        self.df[field + '_num'] = self.df[field]

        # 提取正则表达式和转换值
        expr = rule_map['re']
        value = rule_map['value']

        # 安全地应用正则表达式匹配并转换
        self.df[field + '_num'] = self.df[field].apply(
            lambda x: value if isinstance(x, str) and eval(expr, {'re': re, 'x': x}) else x
        )

        # 替换原字段为转换后的_num字段
        self.df[field] = self.df[field + '_num']
        self.df.drop(columns=[field + '_num'], inplace=True)

        return self.df

    def text_convert_to_dates(self, field: str, regex: str, date_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """
        从指定文本字段中提取日期，并将其转换为指定的格式。

        Args:
            field (str): 需要转换日期的字段名称。
            regex (str): 用于提取日期的正则表达式。
            date_format (str): 日期的目标格式，默认为 '%Y-%m-%d'。

        Returns:
            pd.DataFrame: 转换后包含更新日期格式的DataFrame。
        """
        if field not in self.df.columns:
            print(f"字段 '{field}' 不存在于DataFrame中")
            return self.df

        # 使用正则表达式查找并尝试转换日期
        self.df[field + '_date'] = self.df[field].apply(
            lambda text: convert_date(str(text), regex, date_format) if isinstance(text, str) else text
        )

        # 替换原字段为转换后的_date字段，如果未转换成功则保留原始值
        self.df[field] = self.df[field + '_date']
        self.df.drop(columns=[field + '_date'], inplace=True)  # 删除临时列

        return self.df









if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # 设置显示选项，将浮点数的显示格式设置为完整数值
    pd.set_option('display.float_format', lambda x: '%.0f' % x)

    df = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')
    print(df)
    trans = TypeConversion(df)

    # print(trans.numbers_convert_to_text_bins('age', bins=[0, 20, 40, 60], labels=['young', 'middle', 'old']))
    # print(trans.numbers_convert_to_text_unique('age', mapping={1: 'young', 2: 'middle', 22: 'middle'})['age'])
    # print(trans.date_convert_to_text_bins('join_date', bins=['2024-05-01', '2024-05-06', '2024-05-11'], labels=[1, 2]))
    # print(trans.date_convert_to_timestamp('join_date', timestamp_length=13, date_format='%Y-%m-%d'))

    print(trans.text_convert_to_numbers('text', rule_map={'re': "re.search('test', 'test123456')", 'value': 1}))
    # print(trans.text_convert_to_dates('text',  r'\d{4}-\d{2}-\d{2}'))