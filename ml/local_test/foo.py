import re
import numpy as np
import pandas as pd
from sklearn import datasets

def foo():
    # 加载鸢尾花数据集

    iris_datas = datasets.load_iris()
    iris_df = pd.DataFrame(iris_datas.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    iris_df['label'] = iris_datas.target
    print(iris_df.shape)
    print(iris_df)


if __name__ == '__main__':
    foo()
