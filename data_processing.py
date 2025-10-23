import pandas as pd
from sklearn import preprocessing
import shap
from sklearn.model_selection import train_test_split
import joblib

def read_data(file):
    data = pd.read_excel(file, 0)
    data = data.iloc[:, :]
    return data

def scaler_data_processing(file):
    # 读取了所有数据,格式为dataframe
    data = read_data(file)
    # 读取了所有特征值（X1...Xn）
    raw_feature_value = data.iloc[:, :-1]
    # 读取所有特征值的表头(即Ca，Cl...Leaching pH)
    cols = raw_feature_value.columns
    # 读取了目标值（Y，对应表格数据中的 Immobilization efficiency）
    target_value = data.iloc[:, -1]
    # 将target_value的数据格式转换为dataframe
    target_value = pd.DataFrame(target_value)
    # train_test_split为划分数据集的函数，该函数的返回值为：原始训练特征值；原始测试特征值；训练目标值；测试目标值。这些返回值都是没有经过归一化处理的
    raw_training_feature_value, raw_test_feature_value, training_target_value, test_target_value = train_test_split(raw_feature_value, target_value,
                                                                                            test_size=0.2, random_state=23)
    # 数据归一化，该处采用的是标准归一化方法（X' = (X-μ)/σ）;μ为均值，σ为标准差
    zscore_scaler = preprocessing.StandardScaler()
    # 用训练集计算均值和标准差
    zscore_scaler.fit(raw_training_feature_value)
    # 对raw_training_feature_value进行归一化，原代码是对raw_feature_value进行归一化，raw_feature_value包含了测试集的数据，对raw_feature_value进行归一化会造成数据泄露的问题(这会出现模型效果很好的假象)
    scalar_training_feature_value = zscore_scaler.transform(raw_training_feature_value)
    # 将归一化以后的训练数据转换为dataframe的格式
    scalar_training_feature_value = pd.DataFrame(scalar_training_feature_value, columns=cols)

    # 对raw_training_feature_value进行归一化，归一化时使用的均值和方差是训练集的均值和方差
    scalar_test_feature_value = zscore_scaler.transform(raw_test_feature_value)
    # 将归一化以后的训练数据转换为dataframe的格式
    scalar_test_feature_value = pd.DataFrame(scalar_test_feature_value, columns=cols)

    # 使用SHAP库的K-Means聚类方法，对raw_training_feature_value进行聚类
    raw_feature_train_summary = shap.kmeans(raw_training_feature_value, 10)

    # 保存StandardScaler对象
    joblib.dump(zscore_scaler, 'zscore_scaler.joblib')

    # scaler_data_processing的返回值包括：归一化以后的训练集特征值；训练集目标值；归一化后的测试集特征值；测试集目标值；聚类算法的返回值；原始训练集的特征值，原始测试集的特征值
    return (scalar_training_feature_value, training_target_value, scalar_test_feature_value,
            test_target_value, raw_feature_train_summary,
            raw_training_feature_value,raw_test_feature_value,data)



