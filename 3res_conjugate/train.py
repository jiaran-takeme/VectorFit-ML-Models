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

# -------------------------- 2. 数据加载与预处理（取消偏移，直接标准化） --------------------------
# 读取Excel数据（True工作表，共轭留数场景）
try:
    excel_file = pd.ExcelFile('../Data/S21批量拟合汇总结果(含直流项和比例项).xlsx')
    df = excel_file.parse('True')  # 读取共轭留数的True工作表
    print(f"成功读取True表数据：共{len(df)}条样本，{len(df.columns)}列特征")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# 提取输入（器件参数）和关键输出（3个留数参数：留数1实部/虚部、留数3实部）
X = df[['Cs', 'Rd', 'Rs']].values  # 3个输入特征
# 关键输出定义：y[:,0]=留数1实部，y[:,1]=留数1虚部，y[:,2]=留数3实部（留数3虚部为0，留数2共轭推导）
y = df[['Residue1_Real', 'Residue1_Imag', 'Residue3_Real']].values
print("提取关键留数参数：留数1实部、留数1虚部、留数3实部（取消实部偏移，直接标准化）")

# 划分训练集（90%）和测试集（10%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)
print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")


# -------------------------- 新增：数据集分布可视化 --------------------------
def plot_dataset_distribution(X_train, X_test, y_train, y_test):
    """可视化训练集和测试集的输入特征与输出目标分布"""
    # 输入特征名称和输出目标名称
    input_names = ['Cs (F)', 'Rd (Ω)', 'Rs (Ω)']
    output_names = ['留数1实部', '留数1虚部', '留数3实部']

    # 创建画布（3输入+3输出，共6个子图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练集与测试集分布对比', fontsize=15, y=0.99)

    # 绘制输入特征分布
    for i in range(3):
        ax = axes[0, i]
        # 绘制直方图（训练集蓝色，测试集橙色，半透明叠加）
        ax.hist(X_train[:, i], bins=30, alpha=0.6, color='#2E86AB', label='训练集')
        ax.hist(X_test[:, i], bins=30, alpha=0.6, color='#FFA500', label='测试集')
        ax.set_title(f'{input_names[i]} 分布', fontsize=12)
        ax.set_xlabel(input_names[i], fontsize=10)
        ax.set_ylabel('样本数量', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    # 绘制输出目标分布
    for i in range(3):
        ax = axes[1, i]
        ax.hist(y_train[:, i], bins=30, alpha=0.6, color='#2E86AB', label='训练集')
        ax.hist(y_test[:, i], bins=30, alpha=0.6, color='#FFA500', label='测试集')
        ax.set_title(f'{output_names[i]} 分布', fontsize=12)
        ax.set_xlabel(output_names[i], fontsize=10)
        ax.set_ylabel('样本数量', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 调整标题位置
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 数据集分布对比图已保存为：dataset_distribution.png")


# 调用函数绘制分布对比图
plot_dataset_distribution(X_train, X_test, y_train, y_test)

# -------------------------- 取消留数实部偏移，直接使用原始值 --------------------------
# 分析留数参数分布（直接展示原始值范围）
print(f"\n留数参数分布分析（原始值，未偏移）：")
print(f"  - 留数1实部范围：{np.min(y_train[:, 0]):.6f} ~ {np.max(y_train[:, 0]):.6f}")
print(f"  - 留数1虚部范围：{np.min(y_train[:, 1]):.6f} ~ {np.max(y_train[:, 1]):.6f}（共轭虚部）")
print(f"  - 留数3实部范围：{np.min(y_train[:, 2]):.6f} ~ {np.max(y_train[:, 2]):.6f}（虚部恒为0）")

# 直接使用原始留数数据，不进行偏移处理
y_train_processed = y_train.copy()  # 原始值，无偏移
y_test_processed = y_test.copy()  # 原始值，无偏移
print("已取消实部偏移，直接使用原始留数数据进行标准化")

# -------------------------- 数据标准化（输入+原始留数） --------------------------
# 输入标准化（器件参数）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（直接对原始留数进行标准化，包含实部和虚部）
scaler_res = StandardScaler()
y_train_scaled = scaler_res.fit_transform(y_train_processed)  # 拟合原始训练集
y_test_scaled = scaler_res.transform(y_test_processed)  # 测试集复用训练集参数
print("数据标准化完成：输入和输出均直接标准化（无偏移）")

# 转换为PyTorch张量并创建数据加载器
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"数据加载器创建完成：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 定义神经网络模型（3输入3输出，预测关键留数） --------------------------
class ConjResiduePredictor(nn.Module):
    def __init__(self):
        super(ConjResiduePredictor, self).__init__()
        self.model = nn.Sequential(
            # 输入层→隐藏层1：3→256，批归一化+ReLU
            nn.Linear(3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.0),  # 取消Dropout（可根据过拟合情况调整）

            # 隐藏层1→隐藏层2：256→128，批归一化+ReLU
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # 隐藏层2→隐藏层3：128→64，批归一化+ReLU
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.0),  # 取消Dropout

            # 隐藏层3→隐藏层4：64→32，ReLU
            nn.Linear(64, 32),
            nn.ReLU(),

            # 输出层：32→3（对应3个关键留数）
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置 --------------------------
# 初始化模型、损失函数、优化器
model = ConjResiduePredictor()
criterion = nn.MSELoss()  # 回归任务用均方误差损失
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2正则化（可微调）
)
# 学习率调度器：测试损失停滞10轮则降低学习率（×0.5）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练超参数与记录变量
epochs = 500
best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
train_losses = []  # 记录每轮训练损失
test_losses = []  # 记录每轮测试损失
print(f"\n训练配置完成：总轮次{epochs}，初始学习率0.001，无实部偏移")

# -------------------------- 5. 模型训练（含最佳模型保存） --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印日志，测试损失下降时保存最佳模型）")
print("=" * 80)

for epoch in range(epochs):
    # -------------------------- 训练阶段 --------------------------
    model.train()  # 切换训练模式
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新参数
        train_loss += loss.item() * batch_X.size(0)  # 累加批次损失

    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # -------------------------- 验证阶段 --------------------------
    model.eval()  # 切换评估模式
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        test_losses.append(test_loss)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_conj_residue_predictor_no_offset.pth')
            # 保存训练信息（不含偏移量）
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_res': {'mean': scaler_res.mean_.tolist(), 'std': scaler_res.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr']
            }
            np.save('best_conj_residue_info_no_offset.npy', best_model_info)
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
print("训练结束，加载最佳模型评估（自动推导留数2）")
print("=" * 80)

# 加载最佳模型（无偏移量）
try:
    model.load_state_dict(torch.load('best_conj_residue_predictor_no_offset.pth'))
    best_info = np.load('best_conj_residue_info_no_offset.npy', allow_pickle=True).item()
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 对应训练轮次：Epoch {best_info['epoch']}")
    print(f"   - 最佳测试损失：{best_info['test_loss']:.6f}")
    print(f"   - 标准化器参数已同步加载")
except FileNotFoundError:
    print("⚠️ 警告：未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}")
model.eval()  # 固定评估模式

# 在测试集上预测并反标准化（直接反标准化，无偏移逆操作）
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    # 直接反标准化（无偏移，恢复原始留数）
    y_pred = scaler_res.inverse_transform(y_pred_scaled.numpy())
    y_true = scaler_res.inverse_transform(y_test_scaled)


# -------------------------- 推导完整3个留数（共轭关系不变） --------------------------
def get_complete_residues(pred_key_res, true_key_res):
    """由关键留数参数生成完整3个留数（逻辑不变）"""
    n_samples = len(pred_key_res)
    pred_res = np.zeros((n_samples, 3, 2))  # [实部, 虚部]
    true_res = np.zeros((n_samples, 3, 2))

    # 留数1（预测+真实）
    pred_res[:, 0, 0] = pred_key_res[:, 0]  # 实部
    pred_res[:, 0, 1] = pred_key_res[:, 1]  # 虚部
    true_res[:, 0, 0] = true_key_res[:, 0]
    true_res[:, 0, 1] = true_key_res[:, 1]

    # 留数2（共轭推导）
    pred_res[:, 1, 0] = pred_res[:, 0, 0]  # 实部=留数1实部
    pred_res[:, 1, 1] = -pred_res[:, 0, 1]  # 虚部=-留数1虚部
    true_res[:, 1, 0] = true_res[:, 0, 0]
    true_res[:, 1, 1] = -true_res[:, 0, 1]

    # 留数3（虚部=0）
    pred_res[:, 2, 0] = pred_key_res[:, 2]  # 实部
    pred_res[:, 2, 1] = 0.0
    true_res[:, 2, 0] = true_key_res[:, 2]
    true_res[:, 2, 1] = 0.0

    return pred_res, true_res


# 生成完整留数
pred_residues, true_residues = get_complete_residues(y_pred, y_true)
print(f"\n✅ 已通过共轭关系生成完整3个留数：")
print(f"  - 留数2：实部=留数1实部，虚部=-留数1虚部（共轭）")
print(f"  - 留数3：虚部=0（已知，仅预测实部）")


# -------------------------- 7. 计算评估指标（逻辑不变） --------------------------
def calculate_residue_mae(pred_res, true_res):
    """计算每个留数的实部/虚部MAE"""
    mae_dict = {}
    for i in range(3):
        res_name = f"留数{i + 1}"
        mae_real = np.mean(np.abs(pred_res[:, i, 0] - true_res[:, i, 0]))
        mae_imag = np.mean(np.abs(pred_res[:, i, 1] - true_res[:, i, 1]))
        mae_dict[res_name] = {'实部MAE': mae_real, '虚部MAE': mae_imag}
    return mae_dict


res_mae = calculate_residue_mae(pred_residues, true_residues)
total_mae = (
                    res_mae['留数1']['实部MAE'] + res_mae['留数1']['虚部MAE'] +
                    res_mae['留数2']['实部MAE'] + res_mae['留数2']['虚部MAE'] +
                    res_mae['留数3']['实部MAE'] + res_mae['留数3']['虚部MAE']
            ) / 6

print(f"\n📊 模型评估结果（平均绝对误差MAE）：")
for res, mae in res_mae.items():
    if res == '留数2':
        print(f"   - {res}（共轭推导）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
    elif res == '留数3':
        print(f"   - {res}（虚部=0）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
    else:
        print(f"   - {res}（模型预测）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
print(f"   - 总平均误差：{total_mae:.4f}")

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
plt.title('训练与测试损失曲线（共轭留数）', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('conj_residue_loss_curve_no_offset.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：conj_residue_loss_curve_no_offset.png")

# 8.2 留数对比图
sample_num = min(50, len(pred_residues))
sample_indices = np.arange(sample_num)
fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)  # 3个留数×2（实部/虚部）

for i in range(3):
    res_name = f"留数{i + 1}"
    # 实部对比
    axes[i, 0].plot(sample_indices, true_residues[sample_indices, i, 0], color='#2E86AB', linewidth=2, label='真实实部')
    axes[i, 0].plot(sample_indices, pred_residues[sample_indices, i, 0], color='#FF0000', linewidth=1.5, linestyle='--',
                    label='预测实部')
    axes[i, 0].set_title(f'{res_name} 实部对比（前{sample_num}个样本）', fontsize=12, pad=12)
    axes[i, 0].set_ylabel('实部值', fontsize=10)
    axes[i, 0].legend(fontsize=9)
    axes[i, 0].grid(True, linestyle='--', alpha=0.3)

    # 虚部对比
    axes[i, 1].plot(sample_indices, true_residues[sample_indices, i, 1], color='#2E86AB', linewidth=2, label='真实虚部')
    axes[i, 1].plot(sample_indices, pred_residues[sample_indices, i, 1], color='#FF0000', linewidth=1.5, linestyle='--',
                    label='预测虚部')
    if i == 1:
        axes[i, 1].set_title(f'{res_name} 虚部对比（共轭推导：-留数1虚部）', fontsize=12, pad=12)
    elif i == 2:
        axes[i, 1].set_title(f'{res_name} 虚部对比（固定为0）', fontsize=12, pad=12)
    else:
        axes[i, 1].set_title(f'{res_name} 虚部对比（模型预测）', fontsize=12, pad=12)
    axes[i, 1].set_ylabel('虚部值', fontsize=10)
    axes[i, 1].legend(fontsize=9)
    axes[i, 1].grid(True, linestyle='--', alpha=0.3)

axes[-1, 0].set_xlabel('样本索引', fontsize=10)
axes[-1, 1].set_xlabel('样本索引', fontsize=10)
plt.tight_layout()
plt.savefig('conj_residue_comparison_no_offset.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 共轭留数对比图已保存为：conj_residue_comparison_no_offset.png")

# -------------------------- 9. 打印前10条预测结果 --------------------------
print("\n" + "=" * 120)
print("前10条样本完整预测结果明细（True表：共轭留数）")
print("=" * 120)
print(
    f"{'样本':<6} {'留数1实部(真)':<12} {'留数1实部(预)':<12} {'留数1虚部(真)':<12} {'留数1虚部(预)':<12} "
    f"{'留数2虚部(真)':<12} {'留数2虚部(预)':<12} {'留数3实部(真)':<12} {'留数3实部(预)':<12}"
)
print("-" * 120)
for i in range(min(10, len(pred_residues))):
    print(
        f"{i:<6} {true_residues[i, 0, 0]:<12.4f} {pred_residues[i, 0, 0]:<12.4f} {true_residues[i, 0, 1]:<12.4f} {pred_residues[i, 0, 1]:<12.4f} "
        f"{true_residues[i, 1, 1]:<12.4f} {pred_residues[i, 1, 1]:<12.4f} {true_residues[i, 2, 0]:<12.4f} {pred_residues[i, 2, 0]:<12.4f}"
    )

print("\n" + "=" * 80)
print("所有文件已保存至当前目录：")
print("1. best_conj_residue_predictor_no_offset.pth → 模型参数文件")
print("2. best_conj_residue_info_no_offset.npy → 模型训练信息")
print("3. conj_residue_loss_curve_no_offset.png → 损失曲线")
print("4. conj_residue_comparison_no_offset.png → 留数对比图")
print("5. dataset_distribution.png → 训练集与测试集分布对比图")  # 新增文件
print("=" * 80)