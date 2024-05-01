import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#可视化工龄与合格率关系
# 读取数据
df = pd.read_csv('关系信息.csv')

# 查看数据的基本统计信息
print(df.describe())

# 可视化工龄和合格率的关系
sns.scatterplot(x='工龄', y='合格率', data=df)
plt.title('Relationship between Years of Service and Pass Rate')
plt.xlabel('Years of Service')
plt.ylabel('Pass Rate (%)')
plt.show()

# 计算相关系数，评估工龄和合格率之间的线性关系强度
correlation = df['工龄'].corr(df['合格率'])
print(f"Correlation coefficient: {correlation}")

# 如果需要进一步分析，可以使用回归模型
# 添加常数项，因为线性回归模型需要截距
X = sm.add_constant(df['工龄'])
y = df['合格率']

# 拟合模型
model = sm.OLS(y, X).fit()

# 输出模型的统计摘要
print(model.summary())

# 预测合格率
years_of_service = 4  # 假设我们想知道5年工龄的操作人员的合格率
# 修改这里，使用 numpy 来创建一个正确形状的数组
import numpy as np
predicted_pass_rate = model.predict(sm.add_constant(np.array(years_of_service).reshape(1, -1)))
print(f"Predicted pass rate for {years_of_service} years of service: {predicted_pass_rate[0][1]:.2f}%")