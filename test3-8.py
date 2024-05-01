import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
df = pd.read_csv('M301.csv')
df.iloc[:,[0,1,15]+list(range(26,df.shape[1]))]
print("test")
#原版 有错误
df1 = df.iloc[:,[0,1,15]+list(range(26,df.shape[1]))]
last_records = df1.groupby('日期').last()
last_records = last_records.iloc[:,[2,3]]
last_records['产量'] = last_records['不合格数'] + last_records['合格数']
last_records['合格率'] = last_records['合格数'] / last_records['产量']
last_records['不合格率'] = last_records['不合格数'] / last_records['产量']
print("test")

#创建一个空的DataFrame来存储所有结果
all_faults_df = pd.DataFrame()
dfs = []
for data in tqdm(df1['日期'].unique()):
    #遍历所有列，寻找包含“故障”的列
    faults_data = pd.DataFrame()
    faults_data = pd.concat([faults_data,pd.DataFrame({'日期':[data]})],axis=1)
    for column in df1.columns:
        if '故障' in column:
            #将DataFrame中特定列转换为Panda Series，只考虑日期为3的数据
            series = pd.Series(df1[df1['日期'] == data][column])

            #计算故障开始的位置：非0值开始的位置，并且前一个元素是0
            starts = series[(series != 0)&(series.shift(1) == 0)].index.tolist()

            #计算故障结束的位置：非0值结束的位置，并且后一个元素是0，或者到达最后一个值
            ends = series[(series != 0) & ((series.shift(-1) == 0) | (series.index == series.index[-1]))].index.tolist()
            #计算每次故障持续时间
            durations = [ends[i] - starts[i] +1 for i in range(len(starts))]

            #将结果添加到all_faultsdf DataFrame
            faults_data = pd.concat([faults_data,pd.DataFrame(durations,columns=[f'持续时间_{column}'])],axis=1)
    dfs.append(faults_data)
    print("test")

    all_faults_df = pd.concat(dfs,axis=0,ignore_index=True)
    all_faults_df['日期'].fillna(method='ffill',inplace=True)
    print("test")

    grouped_data = all_faults_df.groupby('日期').mean()
    last_records = pd.concat([last_records,grouped_data],axis=1)
    grouped_data = all_faults_df.groupby('日期').count()

    #定义一个字典，将旧列名映射到新列明
    column_mapping = {
        '持续时间_物料推送装置故障1001': '故障次数_物料推送装置故障1001',
        '持续时间_物料检测装置故障2001': '故障次数_物料检测装置故障2001',
        '持续时间_填装装置检测故障4001': '故障次数_填装装置检测故障4001',
        '持续时间_填装装置定位故障4002': '故障次数_填装装置定位故障4002',
        '持续时间_填装装置填装故障4003': '故障次数_填装装置填装故障4003',
        '持续时间_加盖装置定位故障5001': '故障次数_加盖装置定位故障5001',
        '持续时间_加盖装置加盖故障5002': '故障次数_加盖装置加盖故障5002',
        '持续时间_拧盖装置定位故障6001': '故障次数_拧盖装置定位故障6001',
        '持续时间_拧盖装置拧盖故障6002': '故障次数_拧盖装置拧盖故障6002'
    }
    #使用rename()方法重命名列
    #grouped_data.rename(columns = column_mapping,index=True)
    grouped_data.rename(columns=column_mapping)
    last_records = pd.concat([last_records,grouped_data],axis=1)
    last_records.fillna(0,inplace=True)
    print("test")

    last_records.iloc[:,2:].mean()


    import pandas as pd
    from tqdm import tqdm

    def caculate_mean_fault_duration(file_path):
        df = pd.read_csv(file_path)

        df1 = df.iloc[:, [0, 1, 15] + list(range(26, df.shape[1]))]
        last_records = df1.groupby('日期').last()
        last_records = last_records.iloc[:, [2, 3]]
        last_records['产量'] = last_records['不合格数'] + last_records['合格数']
        last_records['合格率'] = last_records['合格数'] / last_records['产量']
        last_records['不合格率'] = last_records['不合格数'] / last_records['产量']
        # 创建一个空的DataFrame来存储所有结果
        all_faults_df = pd.DataFrame()
        dfs = []
        for data in tqdm(df1['日期'].unique()):
            # 遍历所有列，寻找包含“故障”的列
            faults_data = pd.DataFrame()
            faults_data = pd.concat([faults_data, pd.DataFrame({'日期': [data]})], axis=1)

            for column in df1.columns:
                if '故障' in column:
                    # 将DataFrame中特定列转换为Panda Series，只考虑日期为3的数据
                    series = pd.Series(df1[df1['日期'] == data][column])

                    # 计算故障开始的位置：非0值开始的位置，并且前一个元素是0
                    starts = series[(series != 0) & (series.shift(1) == 0)].index.tolist()

                    # 计算故障结束的位置：非0值结束的位置，并且后一个元素是0，或者到达最后一个值
                    ends = series[
                        (series != 0) & ((series.shift(-1) == 0) | (series.index == series.index[-1]))].index.tolist()
                    # 计算每次故障持续时间
                    durations = [ends[i] - starts[i] + 1 for i in range(len(starts))]

                    # 将结果添加到all_faultsdf DataFrame
                    faults_data = pd.concat([faults_data, pd.DataFrame(durations, columns=[f'持续时间_{column}'])],
                                            axis=1)
            dfs.append(faults_data)
            #print("test")

            all_faults_df = pd.concat(dfs, axis=0, ignore_index=True)
            all_faults_df['日期'].fillna(method='ffill', inplace=True)
            #print("test")

            grouped_data = all_faults_df.groupby('日期').mean()
            last_records = pd.concat([last_records, grouped_data], axis=1)
            grouped_data = all_faults_df.groupby('日期').count()

            # 定义一个字典，将旧列名映射到新列明
            column_mapping = {
                '持续时间_物料推送装置故障1001': '故障次数_物料推送装置故障1001',
                '持续时间_物料检测装置故障2001': '故障次数_物料检测装置故障2001',
                '持续时间_填装装置检测故障4001': '故障次数_填装装置检测故障4001',
                '持续时间_填装装置定位故障4002': '故障次数_填装装置定位故障4002',
                '持续时间_填装装置填装故障4003': '故障次数_填装装置填装故障4003',
                '持续时间_加盖装置定位故障5001': '故障次数_加盖装置定位故障5001',
                '持续时间_加盖装置加盖故障5002': '故障次数_加盖装置加盖故障5002',
                '持续时间_拧盖装置定位故障6001': '故障次数_拧盖装置定位故障6001',
                '持续时间_拧盖装置拧盖故障6002': '故障次数_拧盖装置拧盖故障6002'
            }
            # 使用rename()方法重命名列
            grouped_data.rename(columns=column_mapping, index=True)
            last_records = pd.concat([last_records, grouped_data], axis=1)
            last_records.fillna(0, inplace=True)
            return last_records.iloc[:,2:].mean()



    file_path = "M301.csv"
    result_dfs = [last_records.iloc[:,2:].mean()]
    for i in range(302,311):
        filename = file_path.format(i)
        result = caculate_mean_fault_duration(filename)
        result_dfs.append(result)

    print("test")

    operator_df = pd.concat(result_dfs,axis=1)
    operator_df.T

    operator_info_df = pd.read_excel('附件3/操作人员信息表.xlsx')
    print("test")

    operator_df = operator_df.T
    operator_df['工龄'] = operator_info_df['工龄']
    print("test")

    import matplotlib.font_manager as front_manager

    plt.rcParams['front.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    correlation_matrix = operator_df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=0.5)
    plt.title('correlation_matrix Headmap')
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.show()

    plt.rcParams['front.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    correlation_with_tenure = operator_df.corr()['工龄'].drop('工龄')
    plt.figure(figsize=(10, 6))
    correlation_with_tenure.plot(kind='bar',color='skyblue')
    plt.title('工龄与其他变量相关性')
    plt.xlabel('变量')
    plt.ylabel('相关性')
    plt.xticks(rotation=90)
    plt.show()


