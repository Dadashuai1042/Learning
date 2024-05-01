import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#多项式拟合合格率与工龄关系 并计算各残差值
# 读取数据
df = pd.read_csv('关系信息_.csv')

# 假设 df 是包含工龄和合格率数据的 pandas DataFrame

# 从 DataFrame 中提取工龄作为自变量 x，确保它是一维数组
x = df['工龄'].values

# 提取合格率作为因变量 y
y_pass_rate = df['合格率'].values
y_production = np.array(df['产量'])      # 因变量 (产量)

# 使用 numpy 的 polyfit 方法拟合多项式
degree = 1  # 多项式的次数，这里我们使用二次多项式作为示例
coefficients_pass_rate = np.polyfit(x, y_pass_rate, degree)
coefficients_production = np.polyfit(x, y_production, degree)
# 生成多项式拟合的 x 的值，这里直接使用 np.linspace，它会自动生成一维数组
x_fine = np.linspace(x.min(), x.max(), 100)

# 计算多项式拟合的 y 的值
y_fine_pass_rate = np.polyval(coefficients_pass_rate, x_fine)

y_fine_production = np.polyval(coefficients_production, x_fine)

# 绘图
plt.figure(figsize=(10, 6))

# 合格率与工龄的关系
#plt.subplot(1, 2, 1)
plt.scatter(x, y_pass_rate, color='blue', label='Qualified rate data')
plt.plot(x_fine, y_fine_pass_rate, color='red', label='Linear fit')
plt.title('Relationship between Years of Service and Pass Rate')
plt.xlabel('Years of Service')
plt.ylabel('Pass Rate (%)')
#plt.legend()
''''
# 产量与工龄的关系
plt.subplot(1, 2, 2)
plt.scatter(x, y_production, color='blue', label='产量数据')
plt.plot(x_fine, y_fine_production, color='red', label='二次多项式拟合')
plt.title('产量与工龄的非线性关系')
plt.xlabel('工龄')
plt.ylabel('产量')
plt.legend()
'''
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#工龄与合格率多项式拟合
# 读取数据
df = pd.read_csv('关系信息.csv')  # 确保文件路径正确

# 从 DataFrame 中提取工龄作为自变量 x
x = df['工龄'].values

# 提取合格率作为因变量 y
y_pass_rate = df['合格率'].values

# 使用 numpy 的 polyfit 方法拟合多项式
degree = 1  # 多项式的次数，这里是线性多项式（一次多项式）
coefficients_pass_rate = np.polyfit(x, y_pass_rate, degree)

# 生成预测值
y_fine_pass_rate = np.polyval(coefficients_pass_rate, x)

# 计算残差
residuals_pass_rate = y_pass_rate - y_fine_pass_rate

# 可视化残差
plt.figure(figsize=(10, 6))
plt.scatter(x, residuals_pass_rate, color='blue', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--', label='Fitted Line')
plt.title('Residuals of Linear Fit to Pass Rate')
plt.xlabel('Years of Service')
plt.ylabel('Residuals')
plt.legend()
plt.show()