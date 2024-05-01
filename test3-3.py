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