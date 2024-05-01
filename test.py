#处理持续时间
# 将持续时间开始行保留 其他设置为0 筛选

import pandas as pd

for line_number in range(1, 10):
    # 假设您的Excel文件路径
    excel_file_path = f'result204/result204/result204_break{line_number}.xlsx'

    # 使用pandas读取Excel文件
    df = pd.read_excel(excel_file_path)

    # 检查'时间'列是否已经存在，如果不存在，创建它并填充为0
    if '时间' not in df.columns:
        df['时间'] = 0

    # 初始化持续时间为0
    duration = 0

    # 使用前一次时间的初始值None
    prev_time = 0
    start = 0
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        # 将时间列的值转换为整数
        current_time = int(row['时间'])

        # 检查时间是否连续
        if prev_time ==0 or current_time == prev_time + 1:
            duration += 1  # 时间连续，持续时间加1
        else:
            # 时间不连续，将当前持续时间设置到当前行
            df.at[start, '持续时间'] = duration
            duration = 1  # 重置持续时间为1
            start = index

        # 将计算出的持续时间设置到当前行
        if current_time == (prev_time + 1):
            df.at[index, '持续时间'] = 0
        else:
            df.at[index, '持续时间'] = duration

        # 更新前一次时间为当前时间
        prev_time = current_time
        filtered_df = df[df['持续时间'] != 0]

    # 输出结果
    print(filtered_df)
    #df.to_excel(f'result202_break{line_number}.xlsx')
    filtered_df.to_excel(f'result204_break{line_number}.xlsx')