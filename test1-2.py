import pandas as pd
import xgboost as xgb
import pickle
import openpyxl
#异常检测：XGBoost 还有错误需修改  xgboost导入过慢

# 读取新数据集 M201.csv
data_file = 'M201.csv'
new_data = pd.read_csv(data_file)

# 使用Dask读取所有数据文件
new_data["生产线编号"] = new_data["生产线编号"].str.replace("M201", "0").astype(int)

model = {}
for i in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    with open(f'/home/disk02/ab_design/zrx/test/random_forest/M1{i}_xgboost_model_1000round.pkl', 'rb') as file:
        model[f'{i}'] = pickle.load(file)

# 定义特征和标签，这里假设故障标签列已经转换为适合模型的格式
feature_columns = [col for col in new_data.columns if '故障' not in col]
target_columns = [col for col in new_data.columns if '故障' in col]

# 对新数据集进行预测
X_new = new_data[feature_columns]  # 使用与训练时相同的特征列
dnew = xgb.DMatrix(X_new)
# Predict using the first model
predictions_1 = model['01'].predict(dnew)
predictions_2 = model['02'].predict(dnew)
predictions_3 = model['03'].predict(dnew)
predictions_4 = model['04'].predict(dnew)
predictions_5 = model['05'].predict(dnew)
predictions_6 = model['06'].predict(dnew)
predictions_7 = model['07'].predict(dnew)
predictions_8 = model['08'].predict(dnew)
predictions_9 = model['09'].predict(dnew)
predictions_10 = model['10'].predict(dnew)
predictions = model.predict(dnew)

# Average the predictions
average_predictions = (predictions_1 + predictions_2 + predictions_3 +
                       predictions_4 + predictions_5 + predictions_6 +
                       predictions_7 + predictions_8 + predictions_9 + predictions_10) / 10

# average_predictions = (predictions_1 + predictions_2) / 2
# average_predictions = predictions_1

# 根据预测结果进行故障预测
threshold = 0.5
predicted_faults = (average_predictions > threshold).astype(int)

# print(predicted_faults[:40])

# predicted_1001 = predicted_faults[predicted_faults['物料推送装置故障1001'] == 1]

# predicted_df = pd.DataFrame(predicted_faults[:10000], columns=target_columns)
rows = 1000
predicted_df = pd.DataFrame(predicted_faults)
# predicted_pd.to_excel(f'result201_{rows}_10pklsx1000round.xlsx', index=False, engine='openpyxl')
# print(predicted_pd.keys)  # 0  1  2  3  4  5  6  7  8  9


# 创建一个记录故障的DataFrame
fault_records = pd.DataFrame(columns=['序号', '日期', '开始时间', '持续时长/秒'])

for i in range(1, 11):
    alarm_df = predicted_df[:][predicted_df[i] == 1]  # 根据不同的故障编号筛选数据

    # alarm_df['持续时长/秒'] = alarm_df['时间'].diff().shift(-1).fillna(0)  # 假设持续时长是连续故障之间的时间差
    alarm_df['序号'] = alarm_df.index  # 记录故障编号

    # 记录每段连续时长的起始位置的日期和开始时间
    alarm_df['日期'] = new_data['日期'].iloc[alarm_df.index[0]]  # 使用新数据集中的日期列填充
    alarm_df['开始时间'] = new_data['时间'].iloc[alarm_df.index[0]]  # 使用新数据集中的开始时间列填充
    fault_duration = alarm_df[alarm_df['时间'] > start_time]['时间'].diff().iloc[0]  # 计算故障持续时长
    alarm_df.loc[alarm_df['时间'] == start_time, '持续时长/秒'] = fault_duration.total_seconds()

    # fault_records = fault_records.append(alarm_df[['序号', '日期', '开始时间', '持续时长/秒']], ignore_index=True)

    alarm_df.to_excel(f'result201_break{i}.xlsx', index=False, engine='openpyxl')

# 将预测结果添加到新数据集中
# new_data['Predicted_Fault'] = predicted_faults

# # 保存包含预测结果的新数据集
# new_data.to_csv('M201_predicted.csv', index=False)
