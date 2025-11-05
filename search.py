import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from 三极点预测 import PoleResiduePredictor  # 导入三极点预测模型

# ---------------------- 配置与初始化 ----------------------
# 20Gbps NRZ信号参数（Nyquist频率为10GHz）
NYQUIST_FREQ = 15  # GHz
EVAL_START = 1  # GHz（评估起始频率）
EVAL_END = NYQUIST_FREQ * 1.33  # 13.3GHz（评估终止频率）

# 参数范围（Cs: fF, Rd: Ω, Rs: Ω）
PARAM_RANGES = {
    'Cs': (20, 300),
    'Rd': (50, 800),
    'Rs': (50, 800)
}

# 粒子群优化参数
POP_SIZE = 30  # 粒子数量
MAX_ITER = 200  # 最大迭代次数
W = 0.8  # 惯性权重
C1 = 0.5  # 认知系数
C2 = 0.5  # 社会系数

# 绘图配置（解决中文和负号显示问题）
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False


# ---------------------- S21重构与评估类 ----------------------
class S21Evaluator:
    def __init__(self):
        self.predictor = PoleResiduePredictor()  # 初始化三极点预测模型

        # 1. 生成频率轴（与MATLAB逻辑一致，单位：GHz）
        self.frequency_full = np.linspace(0.1, 100, 1000)  # 0.1-100GHz，1000点
        self.Omega_full = 2 * np.pi * 1j * self.frequency_full  # 复频率（s域）

        # 2. 加载5dB信道曲线（明确列名，Hz转GHz）
        self.channel_freq, self.channel_s21_dB = self._load_channel_data("10dB.csv")

        # 3. 定义评估频段掩码（1~13.3GHz）
        self.eval_mask = (self.frequency_full >= EVAL_START) & (self.frequency_full <= EVAL_END)
        self.eval_freq = self.frequency_full[self.eval_mask]

    def _load_channel_data(self, file_path):
        """加载5dB信道数据（列名：S21 dB20 X为频率(Hz)，S21 dB20 Y为S21值(dB)）"""
        df = pd.read_csv(file_path)

        # 提取频率列（S21 dB20 X）并转换为GHz（Hz -> GHz：除以1e9）
        channel_freq_hz = df['S21 dB20 X'].values
        channel_freq_ghz = channel_freq_hz / 1e9  # 转换单位为GHz

        # 提取S21值列（S21 dB20 Y）
        channel_s21 = df['S21 dB20 Y'].values

        # 插值到CTLE的频率轴（确保与CTLE的频率点对齐）
        interp_func = interp1d(
            channel_freq_ghz,
            channel_s21,
            kind='cubic',
            fill_value="extrapolate"
        )
        return self.frequency_full, interp_func(self.frequency_full)

    def reconstruct_s21(self, poles, residues):
        """复刻MATLAB逻辑重构S21曲线"""
        RS = np.zeros_like(self.frequency_full, dtype=np.complex128)
        dc_term = 0.0 + 0.0j
        scale_term = 0.0 + 0.0j

        for k in range(len(self.frequency_full)):
            pole_residue_sum = np.sum(np.divide(residues, self.Omega_full[k] - poles))
            RS[k] = pole_residue_sum + dc_term + self.Omega_full[k] * scale_term

        return 20 * np.log10(np.abs(RS))  # 转换为dB

    def evaluate(self, cs, rd, rs):
        """评估参数组合：计算补偿后曲线在评估频段的标准差"""
        try:
            # 1. 预测极点和留数
            poles, residues = self.predictor.predict(cs, rd, rs)

            # 2. 重构CTLE的S21曲线（dB）
            ctle_s21_dB = self.reconstruct_s21(poles, residues)

            # 3. 补偿后曲线 = CTLE + 信道（插值后相加）
            compensated_s21 = ctle_s21_dB + self.channel_s21_dB

            # 4. 提取评估频段数据并计算标准差
            eval_data = compensated_s21[self.eval_mask]
            return np.std(eval_data) if len(eval_data) > 0 else float('inf')

        except Exception as e:
            print(f"评估出错: {e}")
            return float('inf')


# ---------------------- 粒子群优化算法 ----------------------
class PSOOptimizer:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.dim = 3  # 参数维度：Cs, Rd, Rs

        # 初始化粒子位置和速度
        self.positions = np.zeros((POP_SIZE, self.dim))
        self.velocities = np.zeros((POP_SIZE, self.dim))

        # 参数范围转换为数组（顺序：Cs, Rd, Rs）
        self.param_min = np.array([PARAM_RANGES['Cs'][0], PARAM_RANGES['Rd'][0], PARAM_RANGES['Rs'][0]])
        self.param_max = np.array([PARAM_RANGES['Cs'][1], PARAM_RANGES['Rd'][1], PARAM_RANGES['Rs'][1]])

        # 初始化位置（在参数范围内随机）
        for i in range(POP_SIZE):
            self.positions[i] = self.param_min + np.random.rand(self.dim) * (self.param_max - self.param_min)

        # 初始化速度（范围内的随机小值）
        self.velocities = 0.1 * (self.param_max - self.param_min) * (np.random.rand(POP_SIZE, self.dim) - 0.5)

        # 个体最优和全局最优
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([evaluator.evaluate(*self._pos_to_params(pos)) for pos in self.positions])
        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]

    def _pos_to_params(self, pos):
        """将粒子位置转换为参数（Cs, Rd, Rs）"""
        return pos[0], pos[1], pos[2]

    def optimize(self):
        """执行粒子群优化"""
        gbest_history = []

        for iter in range(MAX_ITER):
            # 1. 更新速度和位置
            for i in range(POP_SIZE):
                # 速度更新公式
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = C1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social = C2 * r2 * (self.gbest_pos - self.positions[i])
                self.velocities[i] = W * self.velocities[i] + cognitive + social

                # 位置更新并限制在参数范围内
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.param_min, self.param_max)

            # 2. 评估当前位置
            current_vals = np.array([self.evaluator.evaluate(*self._pos_to_params(pos)) for pos in self.positions])

            # 3. 更新个体最优和全局最优
            for i in range(POP_SIZE):
                if current_vals[i] < self.pbest_val[i]:
                    self.pbest_val[i] = current_vals[i]
                    self.pbest_pos[i] = self.positions[i].copy()

            # 更新全局最优
            current_best_idx = np.argmin(current_vals)
            if current_vals[current_best_idx] < self.gbest_val:
                self.gbest_val = current_vals[current_best_idx]
                self.gbest_pos = self.positions[current_best_idx].copy()

            # 记录历史
            gbest_history.append(self.gbest_val)
            print(f"迭代 {iter + 1}/{MAX_ITER} | 最优标准差: {self.gbest_val:.6f} dB")

        return self.gbest_pos, self.gbest_val, gbest_history


# ---------------------- 结果可视化 ----------------------
def plot_results(evaluator, best_params, history):
    cs, rd, rs = best_params
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 绘制优化历史曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history, 'b-', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('评估频段标准差 (dB)')
    plt.title('粒子群优化收敛曲线')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. 绘制补偿前后对比曲线（关键修改：x轴对数坐标 + y轴范围0~-20dB）
    poles, residues = evaluator.predictor.predict(cs, rd, rs)
    ctle_s21 = evaluator.reconstruct_s21(poles, residues)
    compensated_s21 = ctle_s21 + evaluator.channel_s21_dB

    plt.figure(figsize=(10, 6))
    # 绘制三条曲线（保持线型、颜色不变）
    plt.plot(evaluator.frequency_full, evaluator.channel_s21_dB, 'g--', label='原始信道S21', linewidth=1.2)
    plt.plot(evaluator.frequency_full, ctle_s21, 'r-.', label='CTLE S21', linewidth=1.2)
    plt.plot(evaluator.frequency_full, compensated_s21, 'b-', label='补偿后S21', linewidth=1.5)

    # 核心修改1：x轴设为对数坐标（适配频率跨度）
    plt.xscale('log')
    # 核心修改2：y轴范围固定为0~-20dB（确保显示聚焦）
    plt.ylim(-20, 10)
    # 调整x轴范围（聚焦0.1~20GHz，与对数坐标适配）
    plt.xlim(0.1, 100)

    # 标记评估频段（注意：对数坐标下axvspan的x值需与频率范围一致）
    plt.axvspan(EVAL_START, EVAL_END, color='yellow', alpha=0.2, label='评估频段 (1~13.3GHz)')

    # 坐标轴与标题配置（保持不变）
    plt.xlabel('频率 (GHz)', fontsize=12)
    plt.ylabel('S21幅度 (dB)',fontsize=12)
    plt.title(f'最优参数补偿效果 (Cs={cs:.1f}fF, Rd={rd:.1f}Ω, Rs={rs:.1f}Ω)')
    plt.legend()
    plt.grid(alpha=0.3, which='both')  # which='both'：同时显示主/次网格（对数坐标更清晰）
    plt.tight_layout()
    plt.show()


# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    # 初始化评估器和优化器
    evaluator = S21Evaluator()
    optimizer = PSOOptimizer(evaluator)

    # 执行优化
    best_params, best_std, history = optimizer.optimize()

    # 输出最优结果
    print("\n优化完成！最优参数组合：")
    print(f"Cs: {best_params[0]:.2f} fF")
    print(f"Rd: {best_params[1]:.2f} Ω")
    print(f"Rs: {best_params[2]:.2f} Ω")
    print(f"评估频段（1~13.3GHz）标准差：{best_std:.6f} dB")

    # 可视化结果
    plot_results(evaluator, best_params, history)