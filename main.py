import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from joblib import dump
from dask import delayed
from tqdm import dask
#异常检测：RandomForestClassifier随机森林 优化版
#准确率高 运行慢
# 假设数据文件路径已经准备好
data_files = ['M105.csv']  # 示例数据文件

# 使用Dask读取数据文件
ddf = dd.read_csv(data_files)

# 定义特征和标签
feature_columns = [col for col in ddf.columns if '故障' not in col]
target_columns = [col for col in ddf.columns if '故障' in col]

# 分离特征和目标变量
X = ddf[feature_columns]
y = ddf[target_columns]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 定义数值和分类特征
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# 预处理训练和测试数据
X_train_preprocessed = preprocessor.fit_transform(X_train.compute())
X_test_preprocessed = preprocessor.transform(X_test.compute())

# 为每个故障标签单独训练模型
models = {}
for target in target_columns:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_preprocessed, y_train[target].compute())
    models[target] = model

    # 使用模型进行预测
    y_pred = model.predict(X_test_preprocessed)

    # 评估模型
    print(f"Classification report for {target}:")
    y_test_target = y_test[target].compute()
    print(classification_report(y_test_target, y_pred))

# 使用 delayed 异步保存所有模型
save_tasks = [delayed(dump)(model, f'{target}_fault_detection_model.pkl') for target, model in models.items()]

# 执行所有保存任务
# 这将异步执行，减少内存压力
dask.compute(*save_tasks)