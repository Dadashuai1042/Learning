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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
#cuda加速模型 随机森林 未优化
# 假设数据文件路径已经准备好
data_files = ['M105.csv']  # 自行改成Mx0x
# data_files = ['M101.csv','M102.csv','M103.csv','M104.csv','M105.csv','M106.csv','M107.csv','M108.csv','M109.csv','M110.csv']

# 使用Dask读取所有数据文件
ddf = dd.read_csv(data_files)

# 定义特征和标签
feature_columns = [col for col in ddf.columns if '故障' not in col]
target_columns = [col for col in ddf.columns if '故障' in col]

# 分离特征和目标变量
X = ddf[feature_columns].compute()
y = ddf[target_columns].compute()

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

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# 构建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义一个简单的随机森林模型
class RandomForest(nn.Module):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 100)
        self.fc2 = nn.Linear(100, len(target_columns))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 初始化模型和优化器
model = RandomForest()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.items)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in tqdm(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs, 1)
    y_pred = predicted.cpu().numpy()

# 打印分类报告
print(classification_report(y_test.values, y_pred))

# 保存模型
torch.save(model.state_dict(), 'M105_model.pth')
