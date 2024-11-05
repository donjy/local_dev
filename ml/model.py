import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import label_binarize
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from sklearn import datasets

import os
import json
import datetime

"""
    特征处理模块
    1. 逻辑回归       ->    logistic_regression
    2. 线性回归       ->    line_regression
"""


def logistic_regression(df: pd.DataFrame, params: dict):
    """
    逻辑回归分类模型
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - split_ratio (int): 训练集和测试集的划分比例, 默认：80
            - penalty_value (float): 正则化参数复杂度，浮点型，取值范围：(0,10E5]，默认值为0.01。
            - max_iter (int): 最大迭代次数，整型，取值范围：[1,10E5]，默认值为100。
            - tolerance (float): 终止迭代的误差界，浮点型，取值范围：[0,1)，默认值为0.01。
            - penalty_type (str): 正则化类型，字符串，取值范围：['l1','l2']，默认值为'l2'。
            - model_save_path (str): 模型保存路径, 默认为'./model_files'。

    示例：
    params = {
              'split_ratio': 80,
              'penalty_value': 0.01,
              'max_iter': 100,
              'tolerance': 0.01,
              'penalty_type': 'l2',
              'model_save_path': './model_files'
        }

    Returns:
        pd.DataFrame: 增加预测结果列： 预测结果标签列和各个类别的概率
        metrics_json: 包含模型评估指标的字典
    """
    # 参数提取与默认值设置
    split_ratio = params.get('split_ratio', 80) / 100
    penalty_value = params.get('penalty_value', 0.01)
    max_iter = params.get('max_iter', 100)
    tolerance = params.get('tolerance', 0.01)
    penalty_type = params.get('penalty_type', 'l2')
    model_save_path = params.get('model_save_path', "./model_files")

    # 参数验证
    if not (0 < penalty_value <= 1e5):
        raise ValueError("Penalty value must be in the range (0, 10E5].")
    if not (1 <= max_iter <= 1e5):
        raise ValueError("Max iterations must be in the range [1, 10E5].")
    if not (0 <= tolerance < 1):
        raise ValueError("Tolerance must be in the range [0, 1).")
    if penalty_type not in ['l1', 'l2']:
        raise ValueError("Penalty type must be 'l1' or 'l2'.")

    # 检查是否存在label列
    label_col = 'label' if 'label' in df.columns else df.columns[-1]

    # 切分数据集
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42)

    # 训练模型
    model = LogisticRegression(penalty=penalty_type, C=1 / penalty_value, max_iter=max_iter, tol=tolerance,
                               multi_class='auto')
    pipeline = make_pmml_pipeline(model)
    pipeline.fit(X_train, y_train)

    # 预测测试集
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    # 评价指标计算
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    }

    # 训练集预测和评估
    y_pred_train = pipeline.predict(X_train)
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_pred_train),
        "Precision": precision_score(y_train, y_pred_train, average='weighted'),
        "Recall": recall_score(y_train, y_pred_train, average='weighted'),
        "F1 Score": f1_score(y_train, y_pred_train, average='weighted'),
    }

    # 如果是二类或多类问题，计算AUC
    if len(np.unique(y)) > 2:
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))
        if y_test_binarized.shape[1] > 1:  # Only calculate AUC if more than one label
            test_metrics["ROC AUC"] = roc_auc_score(y_test_binarized, y_probs, multi_class='ovr')
    elif len(np.unique(y)) == 2:
        test_metrics["ROC AUC"] = roc_auc_score(y_test, y_probs[:, 1])

    # 保存模型
    filename = f"logistic_regression_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pmml"
    full_model_path = os.path.join(model_save_path, filename)
    sklearn2pmml(pipeline, full_model_path, with_repr=True)

    # 总体评估指标
    metrics = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    metrics_json = json.dumps(metrics, indent=4)

    print(f"Model Evaluation Metrics:\n{metrics_json}")
    print(f"Model saved to {full_model_path}")

    # 将预测结果加入DataFrame
    df_result = df.copy()
    df_result['predicted_label'] = pipeline.predict(X)
    for i, class_label in enumerate(pipeline.classes_):
        df_result[f'probability_class_{class_label}'] = pipeline.predict_proba(X)[:, i]

    return df_result, metrics_json


def line_regression(df: pd.DataFrame, params: dict):
    """
    线性回归模型
    Args:
        df (pd.DataFrame): 原始数据
        params (dict): 包含所有必要参数的字典，包括：
            - split_ratio (int): 训练集和测试集的划分比例, 默认：80
            - penalty_value (float): 正则化参数控制机器的复杂度，浮点型，取值范围：(0,10E5]，默认值为0.01。
            - l1_ratio (float): L1正则化比例，取值范围：(0.0-1.0），默认为0.5。
            - max_iter (int): 最大迭代次数，整型，取值范围：[1,10E5]，默认值为100。
            - tolerance (float): 终止迭代的误差界，浮点型，取值范围：[0,1)，默认值为0.01。
            - penalty_type (str): 正则化类型，字符串，取值范围：['l1','l2','l1+l2']，默认值为'l2'。
            - model_save_path (str): 模型保存路径, 默认为'./model_files'。

    示例：
    params = {
              'split_ratio': 80,
              'penalty_value': 0.01,
              'max_iter': 100,
              'tolerance': 0.01,
              'penalty_type': 'l2',
              'model_save_path': './model_files'
        }

    Returns:
        pd.DataFrame: 增加预测结果列： 预测结果列
        metrics_json: 包含模型评估指标的字典

    """

    split_ratio = params.get('split_ratio', 80) / 100
    penalty_value = params.get('penalty_value', 0.01)
    l1_ratio = params.get('l1_ratio', 0.5)  # L1 ratio for ElasticNet
    max_iter = params.get('max_iter', 100)
    tolerance = params.get('tolerance', 0.01)
    penalty_type = params.get('penalty_type', 'l2')
    model_save_path = params.get('model_save_path', "./model_files")

    os.makedirs(model_save_path, exist_ok=True)

    # 检查是否存在label列
    label_col = 'label' if 'label' in df.columns else df.columns[-1]

    # 切分数据集
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state=42)

    if penalty_type == 'l2':
        model = Ridge(alpha=penalty_value, max_iter=max_iter, tol=tolerance)
    elif penalty_type == 'l1':
        model = Lasso(alpha=penalty_value, max_iter=max_iter, tol=tolerance)
    elif penalty_type == 'l1+l2':
        model = ElasticNet(alpha=penalty_value, l1_ratio=l1_ratio, max_iter=max_iter, tol=tolerance)
    else:
        raise ValueError("Unsupported penalty type. Choose 'l1', 'l2', or 'l1+l2'.")

    pipeline = make_pmml_pipeline(model)
    pipeline.fit(X_train, y_train)

    # 测试集预测和评估
    y_pred_test = pipeline.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # 训练集预测和评估
    y_pred_train = pipeline.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    metrics = {
        "train_metrics": {
            "Mean Squared Error": mse_train,
            "R2 Score": r2_train
        },
        "test_metrics": {
            "Mean Squared Error": mse_test,
            "R2 Score": r2_test
        }
    }

    metrics_json = json.dumps(metrics, indent=4)
    print(metrics_json)

    save_file_name = f"linear_regression_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pmml"
    full_model_path = os.path.join(model_save_path, save_file_name)
    sklearn2pmml(pipeline, full_model_path, with_repr=True)
    print(f"Model saved to {full_model_path}")

    df_result = df.copy()
    df_result['predicted_label'] = pipeline.predict(X)

    return df_result, metrics_json


if __name__ == '__main__':

    # params = {
    #     'split_ratio': 80,
    #     'penalty_value': 0.01,
    #     'max_iter': 100,
    #     'tolerance': 0.01,
    #     'penalty_type': 'l2',
    # }
    #
    # iris_datas = datasets.load_iris()
    # iris_df = pd.DataFrame(iris_datas.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    # iris_df['label'] = iris_datas.target

    # lr_model, metrics_json = logistic_regression(iris_df, params)
    # print(lr_model)

    # 参数配置
    params = {
        'split_ratio': 80,
        'penalty_value': 0.01,
        'max_iter': 100,
        'tolerance': 0.01,
        'penalty_type': 'l1+l2',
        'l1_ratio': 0.6,
        'model_save_path': './model_files'
    }
    # 使用sklearn的datasets加载diabetes数据集进行测试
    data = datasets.load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    df = X.copy()
    df['label'] = y

    # 调用函数
    df_result, metrics_json = line_regression(df, params)
    print(df_result)