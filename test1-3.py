from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

#异常检测：离群点检测 准确率过低

# 读取数据
data = pd.read_csv('F:\LearningResource\比赛\数据挖掘\A题-示例数据\附件1\M101.csv')
# 需要把所有的生产线编号前面的M给删掉
data["生产线编号"] = data["生产线编号"].str.replace("M", "").astype(int)

# 抽样处理，这里以10%为例
data_sampled = data.sample(frac=0.5, random_state=42)  # 可以根据需要调整frac的值

before_columns = [col for col in data_sampled.columns if '故障' not in col]
# 选择需要进行故障检测的装置故障列
fault_columns = [col for col in data_sampled.columns if '故障' in col]

for i, fault_column in enumerate(fault_columns):
    # 为每个故障列生成真实标签
    true_label = data_sampled[fault_column].apply(lambda x: 1 if x > 0 else 0)

    # 计算异常点的比例
    value_counts = true_label.value_counts()
    count_0 = value_counts.get(0, 0)
    count_1 = value_counts.get(1, 0)
    contamination = count_1 / count_0 if count_0 != 0 else 0.0

    # 准备特征数据和目标标签
    X_columns = data_sampled.drop(fault_columns, axis=1)
    y_columns = true_label

    # 训练LocalOutlierFactor模型
    model = LocalOutlierFactor(contamination=contamination)
    labels = model.fit_predict(X_columns)

    # 将异常检测的结果添加到数据集中
    data_sampled[f'anomaly_label_{i}'] = labels

    # 使用 ROC-AUC 评估模型性能
    roc_auc = roc_auc_score(true_label, labels)

    print(f"ROC-AUC for {fault_column}: {roc_auc}")

    cm = confusion_matrix(true_label, labels)

    # 计算精确度、召回率和F1分数
    precision = precision_score(true_label, labels, average='weighted')
    recall = recall_score(true_label, labels, average='weighted')
    f1 = f1_score(true_label, labels, average='weighted')

    print("混淆矩阵：")
    print(cm)
    print("精确度：", precision)
    print("召回率：", recall)
    print("F1分数：", f1)
    print()