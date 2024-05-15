from sklearn.preprocessing import LabelEncoder


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

def fill_nan_data(df):
    """
    预处理数据：使用后向填充，如果失败则填充为0。
    """
    # 使用后向填充（bfill）
    df = df.bfill()
    # 仍然存在NaN的情况下用均值填充
    df = df.fillna(df.mean())
    return df
