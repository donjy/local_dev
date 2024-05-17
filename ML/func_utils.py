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
        将指定字段的文本内容按照用户定义的正则表达式转换为数字。

        Args:
            field (str): 需要转换的字段名称。
            rule_map (dict): 包含正则表达式和对应的数字。
                - unique_value_list (list): 需要转换的唯一值列表。
                - re (str): 正则表达式。
                - value (int): 转换后的值。
             示例:
            {'unique_value_list': ['123', '456', '789'],
             're': r"^\d{3}$",  # 完全匹配三位数字
             'value': 1}

        Returns:
            pd.DataFrame: 转换后的DataFrame。
        """
        unique_value_list = rule_map['unique_value_list']
        regex = rule_map['re']
        value = rule_map['value']

        # 将df的field字段类型转为字符串
        self.df[field] = self.df[field].astype(str)

        # 为每个唯一值应用正则表达式并根据结果修改值
        for unique_value in unique_value_list:
            # 检查当前值是否完全匹配正则表达式
            if re.fullmatch(regex, unique_value):
                # 将符合条件的行的值改为指定的 'value'
                self.df.loc[self.df[field].astype(str) == unique_value, field] = value

        return self.df

    def text_convert_to_dates(self, field: str, rule_map: dict) -> pd.DataFrame:
        """
        从指定文本字段中提取日期，并将其转换为指定的格式。

        Args:
            field (str): 需要转换的字段名称。
            rule_map (dict): 包含正则表达式和日期格式。
                - unique_value_list (list): 需要转换的唯一值列表。
                - re (str): 正则表达式，用于提取日期。
                - date_format (str): 日期的目标格式，默认为 '%Y-%m-%d'。

        Returns:
            pd.DataFrame: 转换后包含更新日期格式的DataFrame。
        """
        unique_value_list = rule_map['unique_value_list']
        regex = rule_map['re']
        date_format = rule_map.get('date_format', '%Y-%m-%d')  # 使用默认格式或提供的格式

        # 为每个唯一值应用正则表达式并根据结果修改值
        for unique_value in unique_value_list:
            # 检查当前值是否匹配正则表达式，并提取日期
            match = re.search(regex, unique_value)
            if match:
                try:
                    # 将匹配到的日期字符串转换为日期对象，然后格式化为指定的格式
                    date_obj = datetime.strptime(match.group(), '%m/%d/%Y')  # 假设提取的日期格式为 MM/DD/YYYY
                    formatted_date = date_obj.strftime(date_format)
                    # 更新DataFrame中匹配到的值
                    self.df.loc[self.df[field] == unique_value, field] = formatted_date
                except ValueError:
                    print(f"日期格式转换错误: {match.group()} 无法按照预期格式解析")
            else:
                # 如果没有匹配到，保留原始值
                continue

        return self.df









if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # 设置显示选项，将浮点数的显示格式设置为完整数值
    pd.set_option('display.float_format', lambda x: '%.0f' % x)

    df = pd.read_excel("E:\\datasets\\test_data.xlsx", sheet_name='test1')

    data = {
        'text': ['123', 'hello world', '999', 'another test example', 'this is run', '123', '456', '789']
    }
    df2 = pd.DataFrame(data)

    # print(df)
    trans = TypeConversion(df)