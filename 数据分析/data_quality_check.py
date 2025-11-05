import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告


# -------------------------- 1. 工具函数：设置中文字体（微软雅黑优先） --------------------------
def set_chinese_font():
    try:
        # 优先加载微软雅黑（对数学符号和小字体支持更好）
        fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    except:
        try:
            # 备用：加载黑体
            fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        except:
            # 兜底：使用系统默认支持字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题


set_chinese_font()

# -------------------------- 2. 数据加载与预处理（Cs优化+极点留数一一对应输出） --------------------------
# 读取Excel数据（需确保文件路径正确）
try:
    excel_file = pd.ExcelFile('../S21批量拟合汇总结果(含直流项和比例项)new.xlsx')
    df = excel_file.parse('False')  # 读取非共轭数据（仅实部）
    print(f"成功读取数据：共{len(df)}条样本，{len(df.columns)}列特征")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# -------------------------- Cs预处理与特征工程（放大小尺度差异） --------------------------
# 1. Cs对数变换（解决1e-14量级尺度压制问题）
df['Cs_log'] = np.log10(df['Cs'] + 1e-16)  # +1e-16避免log(0)
# 2. 构造物理意义交互特征（增强与极点/留数的关联）
df['Cs_Rd'] = df['Cs'] * df['Rd']  # 电容-电阻时间常数（RC）
df['Cs_Rs'] = df['Cs'] * df['Rs']  # 串联RC项
df['Rd_over_Rs'] = df['Rd'] / (df['Rs'] + 1e-8)  # 电阻比值（防除零）

# 提取输入特征（6维：原始Cs+变换特征+交互特征）
input_cols = ['Cs', 'Cs_log', 'Rd', 'Rs', 'Cs_Rd', 'Rd_over_Rs']
X = df[input_cols].values
print(f"输入特征：{input_cols}（共{len(input_cols)}维，已优化Cs尺度）")

# -------------------------- 输出设置：极点1-3 + 留数1-3（一一对应，不排序） --------------------------
# 按原始顺序提取，确保极点i与留数i对应（符合向量拟合配对关系）
output_cols = ['Pole1_Real', 'Pole2_Real', 'Pole3_Real',
               'Residue1_Real', 'Residue2_Real', 'Residue3_Real']
y = df[output_cols].values
print(f"输出特征：{output_cols}（共{len(output_cols)}维，极点-留数一一对应）")

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16, shuffle=True
)
print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

# -------------------------- 输出偏移处理（仅对负极点，留数不偏移） --------------------------
# 分离输出中的“极点”（前3列，均为负）和“留数”（后3列，正负不定）
y_train_poles = y_train[:, :3]  # 训练集极点1-3
y_train_residues = y_train[:, 3:]  # 训练集留数1-3
y_test_poles = y_test[:, :3]  # 测试集极点1-3
y_test_residues = y_test[:, 3:]  # 测试集留数1-3

# 计算极点偏移量（仅用训练集，避免数据泄露）
y_train_pole_min = np.min(y_train_poles)
output_offset = abs(y_train_pole_min) + abs(y_train_pole_min) * 0.1  # 加10%余量防零
print(f"\n极点分布：训练集最小值={y_train_pole_min:.6f}（均为负），偏移量={output_offset:.6f}")

# 仅对极点执行偏移（留数保持原始值，避免破坏物理意义）
y_train_poles_offset = y_train_poles + output_offset
y_test_poles_offset = y_test_poles + output_offset
print(f"偏移后训练集极点范围：{np.min(y_train_poles_offset):.6f} ~ {np.max(y_train_poles_offset):.6f}（均为正）")

# 合并偏移极点和原始留数，作为最终训练输出
y_train_final = np.hstack([y_train_poles_offset, y_train_residues])
y_test_final = np.hstack([y_test_poles_offset, y_test_residues])

# -------------------------- 数据标准化（输入+输出） --------------------------
# 输入标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（偏移极点+原始留数整体标准化）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_final)
y_test_scaled = scaler_y.transform(y_test_final)
print("数据标准化完成：输入6维特征，输出6维（偏移极点+原始留数）")

# 转换为PyTorch张量并创建数据加载器
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"数据加载器：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 定义神经网络模型（输出6维：3极点+3留数） --------------------------
class PoleResiduePredictor(nn.Module):
    def __init__(self):
        super(PoleResiduePredictor, self).__init__()
        self.model = nn.Sequential(
            # 输入层→隐藏层1：6维→256维（充分提取特征）
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 隐藏层1→隐藏层2：256→128（逐步压缩）
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # 隐藏层2→隐藏层3：128→64（进一步压缩）
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 隐藏层3→隐藏层4：64→32（接近输出维度）
            nn.Linear(64, 32),
            nn.ReLU(),

            # 输出层：32→6（极点1-3偏移后 + 留数1-3原始，一一对应）
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置 --------------------------
model = PoleResiduePredictor()
criterion = nn.MSELoss()  # 回归任务用均方误差损失
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2正则化防过拟合
)
# 学习率调度器：测试损失停滞10轮降为50%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练超参数
epochs = 1000
best_test_loss = float('inf')
train_losses = []  # 训练损失记录
test_losses = []  # 测试损失记录
print(f"\n训练配置：总轮次{epochs}，初始学习率0.001，优化器Adam，极点偏移量{output_offset:.6f}")

# -------------------------- 5. 模型训练（含最佳模型保存） --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印日志，测试损失下降时保存模型）")
print("=" * 80)

for epoch in range(epochs):
    # -------------------------- 训练阶段 --------------------------
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss += loss.item() * batch_X.size(0)  # 累加批次损失

    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # -------------------------- 验证阶段 --------------------------
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        test_losses.append(test_loss)

        # 保存最佳模型（记录偏移量和标准化参数，便于后续预测）
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # 保存模型参数
            torch.save(model.state_dict(), 'best_pole_residue_predictor.pth')
            # 保存训练关键信息
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'output_offset': output_offset,
                'input_cols': input_cols,
                'output_cols': output_cols,
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_y': {'mean': scaler_y.mean_.tolist(), 'std': scaler_y.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr']
            }
            np.save('best_model_info.npy', best_model_info)
            print(f"Epoch {epoch + 1:4d}: 测试损失{test_loss:.6f}（历史最佳）→ 保存模型")

    # 学习率调度
    scheduler.step(test_loss)

    # 每10轮打印训练日志
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f} | 学习率: {current_lr:.6f}")

# -------------------------- 6. 加载最佳模型并评估 --------------------------
print("\n" + "=" * 80)
print("训练结束，加载最佳模型评估")
print("=" * 80)

# 加载模型与关键信息
output_offset_loaded = output_offset
try:
    model.load_state_dict(torch.load('best_pole_residue_predictor.pth'))
    best_info = np.load('best_model_info.npy', allow_pickle=True).item()
    output_offset_loaded = best_info['output_offset']
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 训练轮次：Epoch {best_info['epoch']} | 最佳测试损失：{best_info['test_loss']:.6f}")
    print(f"   - 复用极点偏移量：{output_offset_loaded:.6f}")
except FileNotFoundError:
    print("⚠️ 未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}，使用当前偏移量{output_offset_loaded:.6f}")
model.eval()

# 预测并反推原始值（分离极点和留数，仅极点反偏移）
with torch.no_grad():
    # 反标准化：恢复到“偏移极点+原始留数”尺度
    y_pred_scaled = model(X_test_tensor)
    y_pred_final = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true_final = scaler_y.inverse_transform(y_test_scaled)

    # 分离极点和留数，极点反偏移（减偏移量）
    y_pred_poles = y_pred_final[:, :3] - output_offset_loaded  # 原始极点（负）
    y_pred_residues = y_pred_final[:, 3:]  # 原始留数
    y_true_poles = y_true_final[:, :3] - output_offset_loaded  # 真实极点
    y_true_residues = y_true_final[:, 3:]  # 真实留数

# -------------------------- 7. 计算评估指标（MAE） --------------------------
# 极点MAE（一一对应）
mae_p1 = np.mean(np.abs(y_pred_poles[:, 0] - y_true_poles[:, 0]))
mae_p2 = np.mean(np.abs(y_pred_poles[:, 1] - y_true_poles[:, 1]))
mae_p3 = np.mean(np.abs(y_pred_poles[:, 2] - y_true_poles[:, 2]))
# 留数MAE（一一对应）
mae_r1 = np.mean(np.abs(y_pred_residues[:, 0] - y_true_residues[:, 0]))
mae_r2 = np.mean(np.abs(y_pred_residues[:, 1] - y_true_residues[:, 1]))
mae_r3 = np.mean(np.abs(y_pred_residues[:, 2] - y_true_residues[:, 2]))
# 总平均MAE
total_mae = (mae_p1 + mae_p2 + mae_p3 + mae_r1 + mae_r2 + mae_r3) / 6

# 打印评估结果
print(f"\n📊 模型评估结果（平均绝对误差MAE）：")
print("=" * 60)
print(f"{'类型':<8} {'第1个':<12} {'第2个':<12} {'第3个':<12} {'子项平均':<12}")
print("=" * 60)
print(f"{'极点':<8} {mae_p1:.4f}       {mae_p2:.4f}       {mae_p3:.4f}       {(mae_p1 + mae_p2 + mae_p3) / 3:.4f}")
print(f"{'留数':<8} {mae_r1:.4f}       {mae_r2:.4f}       {mae_r3:.4f}       {(mae_r1 + mae_r2 + mae_r3) / 3:.4f}")
print("=" * 60)
print(f"{'总平均':<8} {'':<12} {'':<12} {'':<12} {total_mae:.4f}")

# -------------------------- 8. 可视化结果 --------------------------
# 8.1 训练/测试损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, color='#2E86AB', linewidth=1.5, label='训练损失')
plt.plot(range(1, epochs + 1), test_losses, color='#FF6B6B', linewidth=1.5, label='测试损失')
# 标注最佳模型轮次
if 'best_info' in locals():
    plt.scatter(best_info['epoch'], best_info['test_loss'], color='red', s=60,
                label=f'最佳模型（Epoch{best_info["epoch"]}）', zorder=5)
plt.xlabel('训练轮次（Epoch）', fontsize=12)
plt.ylabel('损失值（MSE）', fontsize=12)
plt.title('训练与测试损失曲线', fontsize=14, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：loss_curve.png")

# 8.2 极点预测对比图（前50个样本，3个子图）
sample_num = min(50, len(y_true_poles))
sample_indices = np.arange(sample_num)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(sample_indices, y_true_poles[sample_indices, i],
            color='#2E86AB', linewidth=2, label='真实极点')
    ax.plot(sample_indices, y_pred_poles[sample_indices, i],
            color='#FF6B6B', linewidth=1.5, linestyle='--', label='预测极点')
    ax.set_title(f'极点{i+1}预测值与真实值对比（前{sample_num}个样本）', fontsize=13, pad=12)
    ax.set_ylabel('极点值（负实数）', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[-1].set_xlabel('样本索引', fontsize=11)
plt.tight_layout()
plt.savefig('pole_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 极点预测对比图已保存为：pole_prediction_comparison.png")

# 8.3 留数预测对比图（前50个样本，3个子图）
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(sample_indices, y_true_residues[sample_indices, i],
            color='#2E86AB', linewidth=2, label='真实留数')
    ax.plot(sample_indices, y_pred_residues[sample_indices, i],
            color='#FF6B6B', linewidth=1.5, linestyle='--', label='预测留数')
    ax.set_title(f'留数{i+1}预测值与真实值对比（前{sample_num}个样本）', fontsize=13, pad=12)
    ax.set_ylabel('留数值（实数）', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[-1].set_xlabel('样本索引', fontsize=11)
plt.tight_layout()
plt.savefig('residue_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 留数预测对比图已保存为：residue_prediction_comparison.png")

# -------------------------- 9. 打印前10条预测结果明细 --------------------------
print("\n" + "=" * 120)
print("前10条样本预测结果明细（极点-留数一一对应）")
print("=" * 120)
print(
    f"{'样本':<6} {'真实极点1':<12} {'预测极点1':<12} {'真实极点2':<12} {'预测极点2':<12} "
    f"{'真实极点3':<12} {'预测极点3':<12} {'真实留数1':<12} {'预测留数1':<12} "
    f"{'真实留数2':<12} {'预测留数2':<12} {'真实留数3':<12} {'预测留数3':<12}"
)
print("-" * 120)
for i in range(min(10, len(y_pred_poles))):
    print(
        f"{i:<6} {y_true_poles[i,0]:<12.4f} {y_pred_poles[i,0]:<12.4f} "
        f"{y_true_poles[i,1]:<12.4f} {y_pred_poles[i,1]:<12.4f} "
        f"{y_true_poles[i,2]:<12.4f} {y_pred_poles[i,2]:<12.4f} "
        f"{y_true_residues[i,0]:<12.4f} {y_pred_residues[i,0]:<12.4f} "
        f"{y_true_residues[i,1]:<12.4f} {y_pred_residues[i,1]:<12.4f} "
        f"{y_true_residues[i,2]:<12.4f} {y_pred_residues[i,2]:<12.4f}"
    )

print("\n" + "=" * 80)
print("所有文件已保存至当前目录：")
print("1. best_pole_residue_predictor.pth → 最佳模型参数")
print("2. best_model_info.npy → 训练信息（含偏移量+标准化参数）")
print("3. loss_curve.png → 训练/测试损失曲线")
print("4. pole_prediction_comparison.png → 极点预测对比图")
print("5. residue_prediction_comparison.png → 留数预测对比图")
print("=" * 80)