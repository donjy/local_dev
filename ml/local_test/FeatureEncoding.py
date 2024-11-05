#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件       :   FeatureEncoding.py
@说明       :   特征编码-API,包含OneHotEncoder,LabelEncoder,默认删除原有列
@时间       :   2023/12/25 16:44:36
@作者       :   Liu mingyue
@版本       :   1.0
'''
# 特征编码功能
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import json


def API_FeatureEncoding(df, ParamsJson, Onehot_suffix="_onehotEncoder", Label_suffix="_labelEncoder"):
    """
    @params:
       df:            要处理的表,类型csv,excel
       ParamsJson:    参数表,包含字段名,类型,编码方式
       Onehot_suffix: OneHot编码的后缀
       Label_suffix : Label编码的后缀
       参数表 eg   :   Column    Type    EncodeType
                      sex       Char    OneHotEncoder
                      ...
    @return:
       df:            处理后的表
    """
    # 找出需要编码的字段
    # df = pd.read_csv(df)
    # json转换为dataframe
    ParamsJson = json.loads(ParamsJson)
    ParamsJson = pd.DataFrame(ParamsJson)
    # 找出需要编码的字段
    Onehot_list = ParamsJson[ParamsJson['EncodeType'] == "OneHotEncoder"]["Column"].to_list()
    Label_list = ParamsJson[ParamsJson['EncodeType'] == "LabelEncoder"]["Column"].to_list()
    # 独热编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(df[Onehot_list])
    # 更改onehot_encoded的float为int
    onehot_encoded = onehot_encoded.astype(int)
    onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(Onehot_list))

    # 加上OneHot列名后缀
    columns = onehot_encoder.get_feature_names_out(Onehot_list)
    columns = [column + Onehot_suffix for column in columns]
    # 修改列名
    onehot_encoded_df.columns = columns

    # 标签编码
    label_encoder = LabelEncoder()
    for column in Label_list:
        df[column + Label_suffix] = label_encoder.fit_transform(df[column])
    # 合并表结果
    encoded_df = pd.concat([df, onehot_encoded_df], axis=1)

    # 删掉原有列
    encoded_df.drop(Onehot_list, axis=1, inplace=True)
    encoded_df.drop(Label_list, axis=1, inplace=True)

    # 生成时间戳
    import time
    timestamp = str(time.time()).split(".")[0]
    filename = r"f:/tmp/result_{}.csv".format(timestamp)
    encoded_df.to_csv(filename, index=False)
    # 返回结果
    return encoded_df
