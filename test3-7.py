import pandas as pd


#工龄和产量关系 写入关系信息



# 假设CSV文件名为data.csv，并且“合格率”列在CSV中的名字是"pass_rate"
for line_number in range(301, 311):
    csv_filename = f'附件3/M{line_number}.csv'
    # 读取CSV文件
    df = pd.read_csv(csv_filename)
    # 确保“合格率”列是数值类型，如果不是，可能需要转换
    # 例如，如果合格率是以百分比的字符串形式存储的，需要转换为数值
    # df['pass_rate'] = df['pass_rate'].str.replace('%', '').astype(float) / 100
    df['日期'] = df['日期'].astype(int)
    max_data=df['日期'].max()
    print(max_data)
    max_data = max_data  # 举例，实际值根据你的数据来定
    pass_total = 0  # 初始化合格数总和
    unpass_total = 0  # 初始化不合格数总和
    # 遍历每个日期，计算合格数和不合格数的总和
    for date in range(1, max_data + 1):
        # 使用布尔索引找到当前日期的所有行
        filtered_df = df[df['日期'] == date]
        # 计算当前日期的合格数和不合格数的最大值
        # 假设合格数和不合格数在同一天不可能同时是最大值，如果有可能，需要进一步的逻辑来处理
        if not filtered_df.empty:
            pass_total += filtered_df['合格数'].max()
            unpass_total += filtered_df['不合格数'].max()

    print(f"合格数总和: {pass_total}")
    print(f"不合格数总和: {unpass_total}")
    print(f"合格率: {pass_total/(unpass_total+pass_total)}")
    print(f"产量：{pass_total+unpass_total}")
    # 计算“合格率”列的最大值

    production = pass_total + unpass_total
    pass_rate = pass_total / production if production > 0 else 0

    # 创建一个新的DataFrame来存储最终结果
    final_df = pd.DataFrame({
        '生产线编号': [f'M{line_number}'],  # 假设生产线是M301，根据实际情况修改
        '合格数总和': [pass_total],
        '不合格数总和': [unpass_total],
        '合格率': [pass_rate],
        '产量': [production]
    })

    # 将新的DataFrame写入CSV文件
    output_csv_filename = 'output.csv'
    final_df.to_csv(output_csv_filename, index=False)

    print(f"结果已写入 {output_csv_filename}")

    # 读取Excel文件和CSV文件
    df1 = pd.read_excel('操作人员信息表.xlsx')
    df2 = pd.read_csv('output.csv')

    # 按'生产线编号'列合并两个DataFrame
    df_combined = pd.merge(df1, df2, on='生产线编号')
    # 将合并后的DataFrame写入新的CSV文件
    #df_combined.to_csv('关系信息.csv', index=False)
    # 将合并后的DataFrame追加写入关系信息CSV文件
    with open('关系信息.csv', 'a', newline='', encoding='utf-8') as f:
        df_combined.to_csv(f, index=False, header=f.tell() == 0)

print("所有数据完成")

