import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
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
    def numbers_convert_to_text_unique(self, field: str, unique_value_lists: list, values: list) -> pd.DataFrame:
        """
        将指定字段的唯一值根据提供的列表转换为指定的多个文本值，未在唯一值列表中的值将保持原值。

        Args:
            field (str): 需要转换的字段名称。
            unique_value_lists (list of list): 多个唯一值列表，每个列表中的值都将转换成对应的文本值。
            values (list): 每个唯一值列表对应的文本值。

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """
        if field in self.df.columns:
            for unique_value_list, value in zip(unique_value_lists, values):
                # 标记应转换的行
                self.df.loc[self.df[field].isin(unique_value_list), field] = value
        else:
            print(f"字段 '{field}' 不存在于DataFrame中")

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
        if field in self.df.columns:
            # 尝试转换时间戳到日期
            def convert_timestamp_to_date(timestamp):
                try:
                    if len(str(timestamp)) == 10:
                        return pd.to_datetime(timestamp, unit='s').strftime(date_format)
                    elif len(str(timestamp)) == 13:
                        return pd.to_datetime(timestamp, unit='ms').strftime(date_format)
                    else:
                        return timestamp  # 保留原始值
                except (ValueError, OverflowError, TypeError):
                    return timestamp  # 如果转换失败，返回原始值

            # 应用转换函数到指定字段
            self.df[field] = self.df[field].apply(convert_timestamp_to_date)
        else:
            print(f"字段 '{field}' 不存在于DataFrame中")

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

    def date_convert_to_text_unique(self, field: str, unique_value_lists: list, values: list,
                                    date_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """
        将指定的日期字段根据给定的多个唯一值列表时间转换为对应的多个文本标签。
        时间格式默认为'%Y-%m-%d'。
        Args:
            field:  (str): 需要转换的日期字段。
            unique_value_lists: (list of list): 多个唯一日期值列表，每个列表中的日期值都将转换成对应的文本值。
            values: (list): 每个唯一值列表对应的文本值。
            date_format: (str): 用于解析和格式化日期的字符串，默认为'%Y-%m-%d'。

        Returns: pd.DataFrame: 转换后的DataFrame。
        """
        if field in self.df.columns:
            # 先将日期字段转换为统一的格式，方便比较
            self.df[field] = pd.to_datetime(self.df[field], errors='coerce', format=date_format).dt.strftime(
                date_format)

            # 为每个唯一日期值列表应用转换
            for unique_dates, value in zip(unique_value_lists, values):
                self.df.loc[self.df[field].isin(unique_dates), field] = value
        else:
            print(f"字段 '{field}' 不存在于DataFrame中")

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
            self.df['_tmp_timestamp'] = pd.to_datetime(self.df[field], format=date_format,
                                                       errors='coerce').dt.tz_localize('Asia/Shanghai',
                                                                                       ambiguous='raise')

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
        将指定字段的文本内容按照用户定义的正则表达式转换为数字。

        Args:
            field (str): 需要转换的字段名称。
            rule_map (dict): 包含多个正则表达式和对应的数字。
                - unique_value_list (list of list): 需要转换的多组唯一值列表。
                - re (list of str): 正则表达式列表。
                - value (list of int): 转换后的值列表。
             示例:
            {'unique_value_list': [['123', '456', '789'],['11','22']],
             're': [r"^\d{3}$", r"^\d{2}$"],
             'value': [2,3]}

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """
        unique_value_lists = rule_map['unique_value_list']
        regexes = rule_map['re']
        values = rule_map['value']

        # 确保字段类型是字符串，以便进行正则匹配
        self.df[field] = self.df[field].astype(str)

        # 为每组正则表达式和值列表应用转换
        for unique_values, regex, value in zip(unique_value_lists, regexes, values):
            for unique_value in unique_values:
                # 检查当前值是否完全匹配正则表达式
                if re.fullmatch(regex, unique_value):
                    # 将符合条件的行的值改为指定的 'value'
                    self.df.loc[self.df[field] == unique_value, field] = value

        return self.df

    def text_convert_to_dates(self, field: str, rule_map: dict) -> pd.DataFrame:
        """
        从指定文本字段中提取日期，并将其转换为指定的格式，支持多种匹配规则。

        Args:
            field (str): 需要转换的字段名称。
            rule_map (dict): 包含多组正则表达式和日期格式的字典。
                - unique_value_lists (list of list): 多个需要转换的唯一值列表。
                - res (list of str): 多个正则表达式，用于提取日期。
                - date_formats (list of str): 多个日期的目标格式，默认为 '%Y-%m-%d'。

        Returns:
            pd.DataFrame: 转换后包含更新日期格式的DataFrame。
        """
        unique_value_lists = rule_map['unique_value_lists']
        regexes = rule_map['res']
        date_formats = rule_map['date_formats']

        # 为每组正则表达式和日期格式应用转换
        for unique_value_list, regex, date_format in zip(unique_value_lists, regexes, date_formats):
            for unique_value in unique_value_list:
                match = re.search(regex, unique_value)
                if match:
                    try:
                        # 将匹配到的日期字符串转换为日期对象，然后格式化为指定的格式
                        date_obj = datetime.strptime(match.group(), date_format)
                        formatted_date = date_obj.strftime(date_format)
                        # 更新DataFrame中匹配到的值
                        self.df.loc[self.df[field] == unique_value, field] = formatted_date
                    except ValueError:
                        print(f"日期格式转换错误: {match.group()} 无法按照预期格式解析")
                else:
                    # 如果没有匹配到，保留原始值
                    continue

        return self.df


# 设置中文字体以防止中文乱码
def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # 设置显示选项，将浮点数的显示格式设置为完整数值
    pd.set_option('display.float_format', lambda x: '%.0f' % x)

    df = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')

