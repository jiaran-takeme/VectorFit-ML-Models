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

# -------------------------- 2. 数据加载与预处理（针对True表共轭极点） --------------------------
# 读取Excel数据（切换为True工作表，需确保文件路径正确）
try:
    excel_file = pd.ExcelFile('../Data/S21批量拟合汇总结果(含直流项和比例项).xlsx')
    df = excel_file.parse('True')  # 读取共轭极点的True工作表
    print(f"成功读取True表数据：共{len(df)}条样本，{len(df.columns)}列特征")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# 提取输入（器件参数）和关键输出（仅需3个参数：极点1实部/虚部、极点3实部）
X = df[['Cs', 'Rd', 'Rs']].values  # 3个输入特征（不变）
# 关键输出：y[:,0]=极点1实部，y[:,1]=极点1虚部，y[:,2]=极点3实部（极点3虚部为0，无需预测）
y = df[['Pole1_Real', 'Pole1_Imag', 'Pole3_Real']].values
print("提取关键输出参数：极点1实部、极点1虚部、极点3实部（极点2由共轭关系推导）")

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16, shuffle=True
)
print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

# -------------------------- 输出自动偏移（针对实部负数值，虚部可能正负，仅偏移实部相关） --------------------------
# 1. 分析输出参数分布（仅关注实部：极点1实部、极点3实部，均可能为负）
pole1_real_min = np.min(y_train[:, 0])  # 极点1实部最小值
pole3_real_min = np.min(y_train[:, 2])  # 极点3实部最小值
y_train_real_min = min(pole1_real_min, pole3_real_min)  # 实部全局最小值（用于偏移）
print(f"\n输出参数分布分析：")
print(f"  - 极点1实部范围：{pole1_real_min:.6f} ~ {np.max(y_train[:, 0]):.6f}")
print(f"  - 极点1虚部范围：{np.min(y_train[:, 1]):.6f} ~ {np.max(y_train[:, 1]):.6f}（共轭虚部）")
print(f"  - 极点3实部范围：{pole3_real_min:.6f} ~ {np.max(y_train[:, 2]):.6f}（虚部为0）")

# 2. 自动计算偏移量（仅对实部参数生效，虚部不偏移，避免破坏共轭关系）
output_offset = abs(y_train_real_min) + abs(y_train_real_min) * 0.1  # 实部偏移量（避开0值）
print(f"自动计算实部偏移量={output_offset:.6f}（实部最小值绝对值+10%余量）")

# 3. 输出偏移处理（仅偏移两个实部参数，虚部保持原样）
y_train_offset = y_train.copy()
y_test_offset = y_test.copy()
y_train_offset[:, 0] += output_offset  # 极点1实部偏移（负→正）
y_train_offset[:, 2] += output_offset  # 极点3实部偏移（负→正）
y_test_offset[:, 0] += output_offset  # 测试集极点1实部偏移
y_test_offset[:, 2] += output_offset  # 测试集极点3实部偏移
print(f"偏移后实部范围：")
print(f"  - 极点1实部：{np.min(y_train_offset[:, 0]):.6f} ~ {np.max(y_train_offset[:, 0]):.6f}（均为正）")
print(f"  - 极点3实部：{np.min(y_train_offset[:, 2]):.6f} ~ {np.max(y_train_offset[:, 2]):.6f}（均为正）")

# -------------------------- 数据标准化（输入+偏移后的输出） --------------------------
# 输入标准化（不变）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（实部已偏移，虚部直接标准化，保持共轭特性）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_offset)  # 拟合偏移后的训练集
y_test_scaled = scaler_y.transform(y_test_offset)  # 测试集用训练集参数
print("数据标准化完成：输入直接标准化，输出实部偏移后标准化、虚部直接标准化")

# 转换为PyTorch张量并创建数据加载器（批量大小64，保持与你原代码一致）
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"数据加载器创建完成：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 定义神经网络模型（仍为3输入3输出） --------------------------
class ConjPolePredictor(nn.Module):
    def __init__(self):
        super(ConjPolePredictor, self).__init__()
        self.model = nn.Sequential(
            # 输入层→隐藏层1：3→256，批归一化+ReLU+Dropout防过拟合
            nn.Linear(3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 隐藏层1→隐藏层2：256→128，批归一化+ReLU
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # 隐藏层2→隐藏层3：128→64，批归一化+ReLU+Dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 隐藏层3→隐藏层4：64→32，ReLU
            nn.Linear(64, 32),
            nn.ReLU(),

            # 输出层：32→3（对应3个关键参数：极点1实部、极点1虚部、极点3实部）
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置（不变） --------------------------
# 初始化模型、损失函数、优化器
model = ConjPolePredictor()
criterion = nn.MSELoss()  # 回归任务用均方误差损失
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2正则化防过拟合
)
# 学习率调度器：测试损失停滞10轮则降低学习率（×0.5）
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练超参数与记录变量
epochs = 1000
best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大
train_losses = []  # 记录每轮训练损失
test_losses = []  # 记录每轮测试损失
print(f"\n训练配置完成：总轮次{epochs}，初始学习率0.001，优化器Adam，实部偏移量{output_offset:.6f}")

# -------------------------- 5. 模型训练（含最佳模型保存，记录偏移量） --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印日志，测试损失下降时保存最佳模型）")
print("=" * 80)

for epoch in range(epochs):
    # -------------------------- 训练阶段 --------------------------
    model.train()  # 切换训练模式（启用Dropout/BatchNorm训练态）
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(batch_X)  # 前向传播（输出：极点1实部/虚部、极点3实部）
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新参数
        train_loss += loss.item() * batch_X.size(0)  # 累加批次损失

    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # -------------------------- 验证阶段（含最佳模型保存） --------------------------
    model.eval()  # 切换评估模式（禁用Dropout/BatchNorm固定）
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        test_losses.append(test_loss)

        # 仅当测试损失下降时，保存最佳模型及训练信息
        if test_loss < best_test_loss:
            best_test_loss = test_loss  # 更新最佳损失
            # 1. 保存模型参数
            torch.save(model.state_dict(), 'best_conj_pole_predictor.pth')
            # 2. 保存训练信息（含偏移量、标准化参数，便于后续预测）
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'output_offset': output_offset,  # 实部偏移量（反标准化需复用）
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_y': {'mean': scaler_y.mean_.tolist(), 'std': scaler_y.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr']
            }
            np.save('best_conj_model_info.npy', best_model_info)
            # 3. 打印保存日志
            print(f"Epoch {epoch + 1:4d}: 测试损失{test_loss:.6f}（历史最佳）→ 保存模型")

    # 学习率调度
    scheduler.step(test_loss)

    # 每10轮打印训练日志
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f} | 当前学习率: {current_lr:.8f}")

# -------------------------- 6. 加载最佳模型并评估（推导完整3个极点） --------------------------
print("\n" + "=" * 80)
print("训练结束，加载最佳模型评估（自动推导极点2）")
print("=" * 80)

# 加载最佳模型（含异常处理，复用偏移量）
output_offset_loaded = output_offset  # 默认用当前偏移量
try:
    # 加载模型参数
    model.load_state_dict(torch.load('best_conj_pole_predictor.pth'))
    # 加载训练信息（含偏移量、标准化参数）
    best_info = np.load('best_conj_model_info.npy', allow_pickle=True).item()
    output_offset_loaded = best_info['output_offset']  # 复用训练时的实部偏移量
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 对应训练轮次：Epoch {best_info['epoch']}")
    print(f"   - 最佳测试损失：{best_info['test_loss']:.6f}")
    print(f"   - 复用实部偏移量：{output_offset_loaded:.6f}")
    print(f"   - 标准化器参数已同步加载")
except FileNotFoundError:
    print("⚠️ 警告：未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}，使用当前计算的偏移量{output_offset_loaded:.6f}")
model.eval()  # 固定评估模式

# 在测试集上预测并反标准化（恢复原始尺度）
with torch.no_grad():
    # 1. 模型预测关键参数（标准化后）
    y_pred_scaled = model(X_test_tensor)
    # 2. 反标准化：恢复到偏移后的尺度
    y_pred_offset = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true_offset = scaler_y.inverse_transform(y_test_scaled)

    # 3. 反偏移：仅实部参数减去偏移量，恢复原始负值
    y_pred = y_pred_offset.copy()
    y_true = y_true_offset.copy()
    y_pred[:, 0] -= output_offset_loaded  # 极点1实部反偏移
    y_pred[:, 2] -= output_offset_loaded  # 极点3实部反偏移
    y_true[:, 0] -= output_offset_loaded  # 真实极点1实部反偏移
    y_true[:, 2] -= output_offset_loaded  # 真实极点3实部反偏移


# -------------------------- 推导完整3个极点（核心：共轭关系生成极点2） --------------------------
def get_complete_poles(pred_key_params, true_key_params):
    """
    由关键参数生成完整3个极点
    pred_key_params: 模型预测的关键参数（n_samples, 3）：[pole1_real, pole1_imag, pole3_real]
    true_key_params: 真实关键参数（n_samples, 3）
    return: 完整预测极点、完整真实极点（n_samples, 3, 2）：[实部, 虚部]
    """
    n_samples = len(pred_key_params)
    # 初始化完整极点数组（每个极点存[实部, 虚部]）
    pred_poles = np.zeros((n_samples, 3, 2))  # pred_poles[i,0] = 极点1，i,1=极点2，i,2=极点3
    true_poles = np.zeros((n_samples, 3, 2))

    # 填充极点1（预测+真实）
    pred_poles[:, 0, 0] = pred_key_params[:, 0]  # 极点1实部
    pred_poles[:, 0, 1] = pred_key_params[:, 1]  # 极点1虚部
    true_poles[:, 0, 0] = true_key_params[:, 0]  # 真实极点1实部
    true_poles[:, 0, 1] = true_key_params[:, 1]  # 真实极点1虚部

    # 生成极点2（共轭关系：实部相同，虚部相反）
    pred_poles[:, 1, 0] = pred_poles[:, 0, 0]  # 极点2实部 = 极点1实部
    pred_poles[:, 1, 1] = -pred_poles[:, 0, 1]  # 极点2虚部 = -极点1虚部
    true_poles[:, 1, 0] = true_poles[:, 0, 0]  # 真实极点2实部 = 真实极点1实部
    true_poles[:, 1, 1] = -true_poles[:, 0, 1]  # 真实极点2虚部 = -真实极点1虚部

    # 填充极点3（虚部为0）
    pred_poles[:, 2, 0] = pred_key_params[:, 2]  # 极点3实部
    pred_poles[:, 2, 1] = 0.0  # 极点3虚部=0（已知）
    true_poles[:, 2, 0] = true_key_params[:, 2]  # 真实极点3实部
    true_poles[:, 2, 1] = 0.0  # 真实极点3虚部=0

    return pred_poles, true_poles


# 生成完整的预测极点和真实极点
pred_poles, true_poles = get_complete_poles(y_pred, y_true)
print(f"\n✅ 已通过共轭关系生成完整3个极点：")
print(f"  - 极点2：实部=极点1实部，虚部=-极点1虚部（共轭）")
print(f"  - 极点3：虚部=0（已知，仅预测实部）")


# -------------------------- 7. 计算评估指标（按完整极点计算误差） --------------------------
# 计算每个极点的实部/虚部平均绝对误差（MAE）
def calculate_pole_mae(pred_poles, true_poles):
    """计算每个极点的实部MAE和虚部MAE"""
    mae_dict = {}
    for i in range(3):
        pole_name = f"极点{i + 1}"
        # 实部MAE
        mae_real = np.mean(np.abs(pred_poles[:, i, 0] - true_poles[:, i, 0]))
        # 虚部MAE（极点3虚部恒为0，误差仅来自预测稳定性）
        mae_imag = np.mean(np.abs(pred_poles[:, i, 1] - true_poles[:, i, 1]))
        mae_dict[pole_name] = {'实部MAE': mae_real, '虚部MAE': mae_imag}
    return mae_dict


pole_mae = calculate_pole_mae(pred_poles, true_poles)
total_mae = (
                    pole_mae['极点1']['实部MAE'] + pole_mae['极点1']['虚部MAE'] +
                    pole_mae['极点2']['实部MAE'] + pole_mae['极点2']['虚部MAE'] +
                    pole_mae['极点3']['实部MAE'] + pole_mae['极点3']['虚部MAE']
            ) / 6  # 总平均误差

print(f"\n📊 模型评估结果（平均绝对误差MAE）：")
for pole, mae in pole_mae.items():
    if pole == '极点2':
        print(f"   - {pole}（共轭推导）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
    elif pole == '极点3':
        print(f"   - {pole}（虚部=0）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
    else:
        print(f"   - {pole}（模型预测）：实部MAE={mae['实部MAE']:.4f}，虚部MAE={mae['虚部MAE']:.4f}")
print(f"   - 总平均误差：{total_mae:.4f}")

# -------------------------- 8. 可视化结果（完整极点对比） --------------------------
# 8.1 绘制训练/测试损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, color='#2E86AB', linewidth=1.5, label='训练损失')
plt.plot(range(1, epochs + 1), test_losses, color='#A23B72', linewidth=1.5, label='测试损失')
# 标注最佳模型对应的epoch
if 'best_info' in locals():
    best_epoch = best_info['epoch']
    best_loss = best_info['test_loss']
    plt.scatter(best_epoch, best_loss, color='red', s=50, zorder=5, label=f'最佳模型（Epoch{best_epoch}）')
plt.xlabel('训练轮次（Epoch）', fontsize=11)
plt.ylabel('损失值（MSE）', fontsize=11)
plt.title('训练与测试损失曲线（True表：共轭极点）', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('conj_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：conj_loss_curve.png")

# 8.2 绘制前50个样本的完整极点对比（分实部/虚部）
sample_num = min(50, len(pred_poles))
sample_indices = np.arange(sample_num)
fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)  # 3个极点×2（实部/虚部）

for i in range(3):
    pole_name = f"极点{i + 1}"
    # 实部对比（第0列子图）
    axes[i, 0].plot(sample_indices, true_poles[sample_indices, i, 0], color='#2E86AB', linewidth=2, label='真实实部')
    axes[i, 0].plot(sample_indices, pred_poles[sample_indices, i, 0], color='#FF0000', linewidth=1.5, linestyle='--',
                    label='预测实部')
    axes[i, 0].set_title(f'{pole_name} 实部对比（前{sample_num}个样本）', fontsize=12, pad=12)
    axes[i, 0].set_ylabel('实部值', fontsize=10)
    axes[i, 0].legend(fontsize=9)
    axes[i, 0].grid(True, linestyle='--', alpha=0.3)

    # 虚部对比（第1列子图）
    axes[i, 1].plot(sample_indices, true_poles[sample_indices, i, 1], color='#2E86AB', linewidth=2, label='真实虚部')
    axes[i, 1].plot(sample_indices, pred_poles[sample_indices, i, 1], color='#FF0000', linewidth=1.5, linestyle='--',
                    label='预测虚部')
    # 标注共轭/固定虚部说明
    if i == 1:
        axes[i, 1].set_title(f'{pole_name} 虚部对比（共轭推导：-极点1虚部）', fontsize=12, pad=12)
    elif i == 2:
        axes[i, 1].set_title(f'{pole_name} 虚部对比（固定为0）', fontsize=12, pad=12)
    else:
        axes[i, 1].set_title(f'{pole_name} 虚部对比（模型预测）', fontsize=12, pad=12)
    axes[i, 1].set_ylabel('虚部值', fontsize=10)
    axes[i, 1].legend(fontsize=9)
    axes[i, 1].grid(True, linestyle='--', alpha=0.3)

axes[-1, 0].set_xlabel('样本索引', fontsize=10)
axes[-1, 1].set_xlabel('样本索引', fontsize=10)
plt.tight_layout()
plt.savefig('conj_pole_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 共轭极点对比图已保存为：conj_pole_comparison.png")

# -------------------------- 9. 打印前10条完整预测结果明细 --------------------------
print("\n" + "=" * 120)
print("前10条样本完整预测结果明细（True表：共轭极点）")
print("=" * 120)
print(
    f"{'样本':<6} {'极点1实部(真)':<12} {'极点1实部(预)':<12} {'极点1虚部(真)':<12} {'极点1虚部(预)':<12} "
    f"{'极点2虚部(真)':<12} {'极点2虚部(预)':<12} {'极点3实部(真)':<12} {'极点3实部(预)':<12}"
)
print("-" * 120)
for i in range(min(10, len(pred_poles))):
    print(
        f"{i:<6} {true_poles[i, 0, 0]:<12.4f} {pred_poles[i, 0, 0]:<12.4f} {true_poles[i, 0, 1]:<12.4f} {pred_poles[i, 0, 1]:<12.4f} "
        f"{true_poles[i, 1, 1]:<12.4f} {pred_poles[i, 1, 1]:<12.4f} {true_poles[i, 2, 0]:<12.4f} {pred_poles[i, 2, 0]:<12.4f}"
    )

print("\n" + "=" * 80)
print("所有文件已保存至当前目录：")
print("1. best_conj_pole_predictor.pth → 共轭极点最佳模型参数文件")
print("2. best_conj_model_info.npy → 模型训练信息（含偏移量+标准化参数）")
print("3. conj_loss_curve.png → 训练/测试损失曲线")
print("4. conj_pole_comparison.png → 共轭极点预测对比图")
print("=" * 80)