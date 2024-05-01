import dask.dataframe as dd
import pandas as pd
from dask_ml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load
import openpyxl

#调用test1的模型预测

# 假设您已经根据问题1的代码训练了模型，并且保存了模型
# 加载模型，这里以故障编号1001为例
model_filename = '物料推送装置故障1001_fault_detection_model.pkl'
model_1001 = load(model_filename)

print("test")
# 问题2的数据文件路径
data_files_2 = 'M201.csv'  # 只有M201

# 使用Dask读取所有数据文件
ddf_2 = dd.read_csv(data_files_2, blocksize=1e6)

print("test")
# 定义特征和标签，这里假设故障标签列已经转换为适合模型的格式
feature_columns_2 = [col for col in ddf_2.columns if '故障' not in col]
target_columns_2 = [col for col in ddf_2.columns if '故障' in col]

# 分离特征和目标变量
X_2 = ddf_2[feature_columns_2]
X_2_df = pd.DataFrame(X_2, columns=feature_columns_2)

print("test")
# 由于我们已经训练了模型，这里直接进行预测
# 使用模型进行预测，这里以故障编号1001为例
y_pred_1001 = model_1001.predict(X_2_df)
print("test")
# 创建一个新DataFrame来保存预测结果
# 假设 '日期', '时间' 是您数据中的列
prediction_df = ddf_2[['日期', '时间']].compute().reset_index(drop=True)
prediction_df['1001故障'] = y_pred_1001

# 根据预测结果生成报警信息
# 这里需要您根据实际情况来实现写入Excel的功能
alarm_df = prediction_df[prediction_df['1001故障'] == 1001]
# 假设 prediction_df 是包含预测结果的DataFrame，并且它包含'时间'和'1001故障'列

# 计算故障的持续时长
# 我们通过比较当前行和下一行来确定故障的结束
# 如果下一行没有故障（即'1001故障'列的值为0或其它非故障标记），则当前行是故障结束
# 否则，故障持续到下一行（因为故障状态相同）

# 初始化一个空的Series来保存持续时长
alarm_df['持续时长/秒'] = pd.Series([0] * len(alarm_df), index=alarm_df.index)

# 遍历故障事件，计算持续时长
for i in range(len(alarm_df) - 1):
    if alarm_df.loc[i, '1001故障'] == 1001:  # 如果当前行是故障
        start_time = alarm_df.loc[i, '时间']  # 故障开始时间
        # 查找下一个非故障行的索引
        try:
            end_index = alarm_df[alarm_df['1001故障'] != 1001].index[0]
        except IndexError:  # 如果故障持续到数据集末尾
            end_index = len(alarm_df)
        end_time = alarm_df.loc[end_index - 1, '时间']  # 故障结束时间
        duration = end_time - start_time  # 计算持续时长
        # 将计算出的持续时长赋值给故障开始行
        alarm_df.loc[i, '持续时长/秒'] = duration

# 注意：这段代码是一个概念性示例，它可能需要根据实际的数据集结构进行调整。
#alarm_df['持续时长/秒'] = alarm_df['时间'].diff().shift(-1).fillna(0)  # 假设持续时长是连续故障之间的时间差

# 将报警信息写入到result2.xlsx中
alarm_df.to_excel('result2.xlsx', index=False, engine='openpyxl')

# 注意：这里只展示了针对一个故障编号的预测和结果生成过程
# 如果有多个故障编号，需要对每个故障编号重复上述过程

