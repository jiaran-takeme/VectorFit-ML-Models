import numpy as np
import matplotlib.pyplot as plt
# 设置字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 1. 参数设置 ----------------------
# 频率范围：0.1GHz到100GHz，间隔0.1GHz
frequency = np.linspace(0.1, 100, 1000)  # 单位：GHz

# 计算复频率 (s = jω = j*2πf)
Omega = 2 * np.pi * 1j * frequency

# 预测的极点和留数
predicted_poles = np.array([
    -720.335018 + 0.000000j,
    -165.244716 + 0.000000j,
    -112.241542 + 0.000000j
], dtype=np.complex128)

predicted_residues = np.array([
    -767.348616 + 0.000000j,
    1252.038779 + 0.000000j,
    -579.123745 + 0.000000j
], dtype=np.complex128)

# 直流项和比例项
dc_term = 0.0 + 0.0j
scale_term = 0.0 + 0.0j

# ---------------------- 2. 用预测的极点留数还原S21 ----------------------
predicted_S21 = np.zeros_like(frequency, dtype=np.complex128)

for k in range(len(frequency)):
    # 核心公式：S21(s) = Σ[留数/(s-极点)] + 直流项 + s*比例项
    sum_term = np.sum(np.divide(predicted_residues, Omega[k] - predicted_poles))
    predicted_S21[k] = sum_term + dc_term + Omega[k] * scale_term

# 转换为dB
predicted_S21_dB = 20 * np.log10(np.abs(predicted_S21)) - 2.8

# ---------------------- 3. 读取CSV中的原始S21数据 ----------------------
# 假设CSV文件格式：第一列是频率(GHz)，第二列是S21(dB)
# 请替换为你的CSV文件路径
raw_data = np.genfromtxt('search.csv', delimiter=',', skip_header=1)  # skip_header=1表示跳过表头
raw_frequency = raw_data[:, 0]  # 原始频率数据
raw_S21_dB = raw_data[:, 1]     # 原始S21数据(dB)

# ---------------------- 4. 绘图对比 ----------------------
plt.figure(figsize=(10, 6))

# 绘制原始S21曲线
plt.plot(frequency, raw_S21_dB,
         color='blue', linestyle='-', linewidth=1.2,
         label='原始S21 (CSV数据)')

# 绘制预测极点留数还原的S21曲线
plt.plot(frequency, predicted_S21_dB,
         color='red', linestyle='--', linewidth=1.2,
         label='预测极点留数还原的S21')

# 图表设置（使用微软雅黑字体）
plt.xlabel('频率 (GHz)', fontsize=12)
plt.ylabel('S21 (dB)', fontsize=12)
plt.title('原始S21与预测极点留数还原S21对比', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0.1, 100])  # 限制频率范围
plt.tight_layout()

# 显示图像
plt.show()