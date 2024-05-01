import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# 假设数据文件路径已经准备好
data_files = ['M105.csv']

# 使用Dask读取所有数据文件
ddf = dd.read_csv(data_files,blocksize=1e6)

# 定义特征和标签
feature_columns = [col for col in ddf.columns if '故障' not in col]
target_columns = [col for col in ddf.columns if '故障' in col]

# 分离特征和目标变量
X = ddf[feature_columns]
y = ddf[target_columns]

# 分割训练集和测试集，添加了shuffle=True参数
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 定义数值和分类特征
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

# 创建数值和分类列的转换器
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 填充缺失值
    ('scaler', StandardScaler())  # 标准化数值特征
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 填充缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 编码分类特征
])

# 使用ColumnTransformer来组合数值和分类转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 为每个故障标签单独训练模型
models = {}
for target in target_columns:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    pipeline.fit(X_train.compute(), y_train[target].compute())  # 确保使用 .compute() 计算训练数据和标签

    # 使用模型进行预测
    y_pred = pipeline.predict(X_test.compute())  # 预测也需要 .compute()

    # 评估模型
    print(f"Classification report for {target}:")
    # 确保 y_test[target] 是一个 NumPy 数组
    y_test_target = y_test[target].compute()
    # 由于 y_pred 已经是 NumPy 数组，我们不需要再次调用 .compute()
    print(classification_report(y_test_target, y_pred))
    # 将训练好的模型存储到models字典中
    models[target] = pipeline  # 这里添加了这一行
    # 保存所有模型
for target, model in models.items():
    model_filename = f'{target}M105_fault_detection_model.pkl'
    dump(model, model_filename)
    print(f"Model saved as {model_filename}")  # 打印保存信息

