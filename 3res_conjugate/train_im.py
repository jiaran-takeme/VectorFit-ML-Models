import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.ao.nn.quantized import Dropout
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')  # 屏蔽无关警告


# -------------------------- 1. 工具函数：设置中文字体 --------------------------
def set_chinese_font():
    try:
        fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    except:
        try:
            fm.fontManager.addfont('/Library/Fonts/Songti.ttc')
            plt.rcParams['font.sans-serif'] = ['Songti SC', 'DejaVu Sans']
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


set_chinese_font()

# -------------------------- 2. 数据加载与预处理（仅保留留数1虚部作为输出） --------------------------
# 读取Excel数据（True工作表，共轭留数场景）
try:
    excel_file = pd.ExcelFile('../S21批量拟合汇总结果(含直流项和比例项).xlsx')
    df = excel_file.parse('True')  # 读取共轭留数的True工作表
    print(f"成功读取True表数据：共{len(df)}条样本，{len(df.columns)}列特征")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# 提取输入（器件参数）和唯一输出（留数1虚部）
X = df[['Cs', 'Rd', 'Rs']].values  # 3个输入特征（不变）
y = df[['Residue1_Imag']].values   # 仅保留“留数1虚部”作为输出（单输出）
print("提取预测目标：仅留数1虚部（单输出）")

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16, shuffle=True
)
print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")


# -------------------------- 数据集分布可视化（含输入和单输出） --------------------------
def plot_dataset_distribution(X_train, X_test, y_train, y_test):
    """可视化训练集和测试集的输入特征与输出目标分布"""
    input_names = ['Cs (F)', 'Rd (Ω)', 'Rs (Ω)']
    output_names = ['留数1虚部']  # 仅单输出

    # 创建画布（3输入+1输出，共4个子图）
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('训练集与测试集分布对比', fontsize=15, y=0.99)

    # 绘制输入特征分布（3个输入）
    for i in range(3):
        ax = axes[i // 2, i % 2]  # 前3个子图
        ax.hist(X_train[:, i], bins=30, alpha=0.6, color='#2E86AB', label='训练集')
        ax.hist(X_test[:, i], bins=30, alpha=0.6, color='#FFA500', label='测试集')
        ax.set_title(f'{input_names[i]} 分布', fontsize=12)
        ax.set_xlabel(input_names[i], fontsize=10)
        ax.set_ylabel('样本数量', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 绘制输出目标分布（留数1虚部）
    ax = axes[1, 1]  # 第4个子图
    ax.hist(y_train, bins=30, alpha=0.6, color='#2E86AB', label='训练集')
    ax.hist(y_test, bins=30, alpha=0.6, color='#FFA500', label='测试集')
    ax.set_title(f'{output_names[0]} 分布', fontsize=12)
    ax.set_xlabel(output_names[0], fontsize=10)
    ax.set_ylabel('样本数量', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 数据集分布对比图已保存为：dataset_distribution.png")


# 调用函数绘制分布对比图
plot_dataset_distribution(X_train, X_test, y_train, y_test)

# -------------------------- 数据标准化（输入+单输出） --------------------------
# 分析留数1虚部分布
print(f"\n留数1虚部分布分析（原始值）：")
print(f"  - 范围：{np.min(y_train):.6f} ~ {np.max(y_train):.6f}")

# 输入标准化（器件参数）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（仅留数1虚部）
scaler_res = StandardScaler()
y_train_scaled = scaler_res.fit_transform(y_train)  # 单输出标准化
y_test_scaled = scaler_res.transform(y_test)
print("数据标准化完成：输入（3特征）和输出（留数1虚部）均标准化")

# 转换为PyTorch张量并创建数据加载器
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"数据加载器创建完成：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 定义神经网络模型（3输入→1输出） --------------------------
class ResidueImagPredictor(nn.Module):
    def __init__(self):
        super(ResidueImagPredictor, self).__init__()
        self.model = nn.Sequential(
            # 输入层→隐藏层1：3→256
            nn.Linear(3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # 隐藏层1→隐藏层2：256→128
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            Dropout(0.1),

            # 隐藏层1→隐藏层2：256→128
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # 隐藏层1→隐藏层2：256→128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),


            # 隐藏层2→隐藏层3：128→64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            Dropout(0.1),

            # 隐藏层3→隐藏层4：64→32
            nn.Linear(64, 32),
            nn.ReLU(),

            # 输出层：32→1（仅预测留数1虚部）
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置 --------------------------
model = ResidueImagPredictor()
criterion = nn.MSELoss()  # 回归任务用均方误差
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2正则化
)
# 学习率调度器：测试损失停滞10轮则降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练超参数与记录变量
epochs = 500
best_test_loss = float('inf')
train_losses = []
test_losses = []
print(f"\n训练配置完成：总轮次{epochs}，输出维度=1（仅留数1虚部）")

# -------------------------- 5. 模型训练 --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印日志，保存最佳模型）")
print("=" * 80)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)  # 输出维度：(batch_size, 1)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # 验证阶段
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        test_losses.append(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_residue1_imag_predictor.pth')
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_res': {'mean': scaler_res.mean_.tolist(), 'std': scaler_res.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr']
            }
            np.save('best_residue1_imag_info.npy', best_model_info)
            print(f"Epoch {epoch + 1:4d}: 测试损失{test_loss:.6f}（历史最佳）→ 保存模型")

    # 学习率调度
    scheduler.step(test_loss)

    # 每10轮打印日志
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f} | 当前学习率: {current_lr:.8f}")

# -------------------------- 6. 加载最佳模型并评估 --------------------------
print("\n" + "=" * 80)
print("训练结束，加载最佳模型评估（仅留数1虚部）")
print("=" * 80)

# 加载最佳模型
try:
    model.load_state_dict(torch.load('best_residue1_imag_predictor.pth'))
    best_info = np.load('best_residue1_imag_info.npy', allow_pickle=True).item()
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 对应训练轮次：Epoch {best_info['epoch']}")
    print(f"   - 最佳测试损失：{best_info['test_loss']:.6f}")
except FileNotFoundError:
    print("⚠️ 警告：未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}")
model.eval()

# 在测试集上预测并反标准化
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_res.inverse_transform(y_pred_scaled.numpy())  # 反标准化为原始虚部值
    y_true = scaler_res.inverse_transform(y_test_scaled)          # 真实虚部值


# -------------------------- 7. 计算评估指标（仅留数1虚部MAE） --------------------------
def calculate_imag_mae(pred_imag, true_imag):
    """计算留数1虚部的平均绝对误差"""
    return np.mean(np.abs(pred_imag - true_imag))


mae_imag = calculate_imag_mae(y_pred, y_true)
print(f"\n📊 模型评估结果：")
print(f"   - 留数1虚部平均绝对误差（MAE）：{mae_imag:.6f}")

# -------------------------- 8. 可视化结果 --------------------------
# 8.1 损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, color='#2E86AB', linewidth=1.5, label='训练损失')
plt.plot(range(1, epochs + 1), test_losses, color='#A23B72', linewidth=1.5, label='测试损失')
if 'best_info' in locals():
    best_epoch = best_info['epoch']
    best_loss = best_info['test_loss']
    plt.scatter(best_epoch, best_loss, color='red', s=50, zorder=5, label=f'最佳模型（Epoch{best_epoch}）')
plt.xlabel('训练轮次（Epoch）', fontsize=11)
plt.ylabel('损失值（MSE）', fontsize=11)
plt.title('训练与测试损失曲线（仅预测留数1虚部）', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('residue1_imag_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：residue1_imag_loss_curve.png")

# 8.2 留数1虚部对比图
sample_num = min(50, len(y_pred))
sample_indices = np.arange(sample_num)
plt.figure(figsize=(12, 6))
plt.plot(sample_indices, mae_imag[sample_indices, 0], color='#2E86AB', linewidth=2, label='真实留数1虚部')
plt.plot(sample_indices, y_pred[sample_indices, 0], color='#FF0000', linewidth=1.5, linestyle='--', label='预测留数1虚部')
plt.xlabel('样本索引', fontsize=11)
plt.ylabel('留数1虚部值', fontsize=11)
plt.title(f'留数1虚部对比（前{sample_num}个测试样本）', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('residue1_imag_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 留数1虚部对比图已保存为：residue1_imag_comparison.png")

# -------------------------- 9. 打印前10条预测结果 --------------------------
print("\n" + "=" * 60)
print("前10条样本预测结果明细（仅留数1虚部）")
print("=" * 60)
print(f"{'样本':<6} {'真实值':<15} {'预测值':<15} {'绝对误差':<10}")
print("-" * 60)
for i in range(min(10, len(y_pred))):
    true_val = mae_imag[i, 0]
    pred_val = y_pred[i, 0]
    abs_err = abs(true_val - pred_val)
    print(f"{i:<6} {true_val:<15.6f} {pred_val:<15.6f} {abs_err:<10.6f}")

print("\n" + "=" * 80)
print("所有文件已保存至当前目录：")
print("1. best_residue1_imag_predictor.pth → 模型参数文件")
print("2. best_residue1_imag_info.npy → 模型训练信息")
print("3. residue1_imag_loss_curve.png → 损失曲线")
print("4. residue1_imag_comparison.png → 留数1虚部对比图")
print("5. dataset_distribution.png → 训练集与测试集分布对比图")
print("=" * 80)