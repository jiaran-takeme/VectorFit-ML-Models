# 导入需要的库：numpy处理数值计算，matplotlib.pyplot负责绘图
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 1. 手动输入已知参数（与MATLAB完全对应，GHz单位） ----------------------
# 1.1 频率范围：0.1GHz到100GHz，间隔0.1GHz（共1000个点，和MATLAB的linspace效果一致）
frequency = np.linspace(0.1, 100, 1000)  # 单位：GHz，shape为(1000,)

# 1.2 计算复频率Omega（s域变量：s = jω = j*2πf，基于GHz单位，与你的需求一致）
# np.complex128确保复数精度，1j对应MATLAB的1i
Omega = 2 * np.pi * 1j * frequency

# 1.3 已知的3个极点（纯实数，与MATLAB参数完全相同）
pole = np.array([
    -636.403044998463 + 0.000000e+00j,  # a_1
    -632.021718270331 + 0.000000e+00j,  # a_2
    -220.361689771102 + 0.000000e+00j   # a_3
], dtype=np.complex128)  # 指定复数类型，避免精度问题

# 1.4 已知的3个留数（纯实数，与MATLAB参数完全相同）
residue = np.array([
    -78787.6765249793 + 0.000000e+00j,  # c_1
    78447.5038937009 + 0.000000e+00j,   # c_2
    173.251009745321 + 0.000000e+00j    # c_3
], dtype=np.complex128)


# 1.3 已知的3个极点（纯实数，与MATLAB参数完全相同）
pole2 = np.array([
    -636.403044998463 + 0.000000e+00j,  # a_1
    -632.021718270331 + 0.000000e+00j,  # a_2
    -220.361689771102 + 0.000000e+00j   # a_3
], dtype=np.complex128)  # 指定复数类型，避免精度问题

# 1.4 已知的3个留数（纯实数，与MATLAB参数完全相同）
residue2 = np.array([
    -78787.6765249793 + 0.000000e+00j,  # c_1
    78447.5038937009 + 0.000000e+00j,   # c_2
    173.251009745321 + 0.000000e+00j    # c_3
], dtype=np.complex128)
# 1.5 直流项和比例项（均为0，与MATLAB一致）
dc_term = 0.000000e+00 + 0.000000e+00j    # 直流项d
scale_term = 0.000000e+00 + 0.000000e+00j # 比例项e

# ---------------------- 2. 用极点留数重构S21（复刻MATLAB逻辑） ----------------------
# 初始化存储重构结果的数组（复数类型，shape与frequency一致）
RS = np.zeros_like(frequency, dtype=np.complex128)
RS2 = np.zeros_like(frequency, dtype=np.complex128)
# 遍历每个频率点，计算重构的S21（对应MATLAB的for循环）
for k in range(len(frequency)):
    # 向量拟合核心公式：S21(s) = Σ[留数/(s-极点)] + 直流项 + s*比例项
    # 逐元素除法用np.divide，求和用np.sum，完全对应MATLAB的sum(residue./(Omega(k)-pole))
    pole_residue_sum = np.sum(np.divide(residue, Omega[k] - pole))
    pole_residue_sum2 = np.sum(np.divide(residue2, Omega[k] - pole2))
    # 累加直流项和比例项，得到当前频率点的S21值
    RS[k] = pole_residue_sum + dc_term + Omega[k] * scale_term
    RS2[k] = pole_residue_sum2 + dc_term + Omega[k] * scale_term

# ---------------------- 3. 绘制拟合曲线图（复刻MATLAB绘图样式） ----------------------
# 创建图形窗口：设置窗口名、大小（对应MATLAB的figure('Name',..., 'Position',...)）
plt.figure('图2：极点留数重构拟合结果', figsize=(8, 5))  # figsize=(宽,高)，单位英寸

# 绘制重构的S21幅度曲线（dB形式：20*log10(abs(复数))，蓝色实线，线宽1.2）
# np.abs计算复数模，np.log10计算对数，完全对应MATLAB的20*log10(abs(RS))
plt.plot(
    frequency,
    20 * np.log10(np.abs(RS)),
    color='blue',
    linestyle='-',
    linewidth=1.2,
    label='S21 fitted (original)'  # 曲线标签，对应MATLAB的legend
)

plt.plot(
    frequency,
    20 * np.log10(np.abs(RS2)),
    color='red',
    linestyle='--',
    linewidth=1.2,
    label='S21 fitted (pole-residue)'  # 曲线标签，对应MATLAB的legend
)

# 设置坐标轴标签（对应MATLAB的xlabel、ylabel）
plt.xlabel('Frequency (GHz)', fontsize=11)
plt.ylabel('S21 Magnitude (dB)', fontsize=11)

# 设置图例（对应MATLAB的legend）
plt.legend(fontsize=10)

# 显示网格（对应MATLAB的grid on）
plt.grid(True, linestyle='-', alpha=0.3)  # alpha控制网格透明度，更美观

# 调整布局（避免标签被截断，对应MATLAB的tightlayout）
plt.tight_layout()

# 显示图形（对应MATLAB的绘图窗口弹出）
plt.show()