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

# -------------------------- 2. 数据加载与预处理（核心修改：删除极点排序） --------------------------
# 读取Excel数据（需确保文件路径正确）
try:
    excel_file = pd.ExcelFile('../Data/S21批量拟合汇总结果(含直流项和比例项).xlsx')
    df = excel_file.parse('False')  # 读取非共轭极点的False工作表
    print(f"成功读取数据：共{len(df)}条样本，{len(df.columns)}列特征")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# 提取输入（器件参数）和输出（实极点）
# 核心修改：直接提取Excel中的Pole1_Real/Pole2_Real/Pole3_Real，不做任何排序
X = df[['Cs', 'Rd', 'Rs']].values  # 3个输入特征
y = df[['Pole1_Real', 'Pole2_Real', 'Pole3_Real']].values  # 保持Excel列顺序，不排序
print("输出极点保持与Excel表格一致的顺序（Pole1_Real→Pole2_Real→Pole3_Real），未做排序")

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=16, shuffle=True
)
print(f"数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

# -------------------------- 新增：输出自动偏移（针对均为负极点） --------------------------
# 1. 计算训练集极点最小值（仅用训练集，避免数据泄露）
y_train_min = np.min(y_train)
print(f"\n极点分布分析：训练集极点最小值={y_train_min:.6f}，最大值={np.max(y_train):.6f}（均为负）")

# 2. 自动计算偏移量：最小值绝对值 + 10%余量（确保偏移后所有值为正，且远离0）
output_offset = abs(y_train_min) + abs(y_train_min) * 0.1  # 自动适配数据，无需手动调参
print(f"自动计算偏移量={output_offset:.6f}（最小值绝对值+10%余量）")

# 3. 极点偏移：所有输出加偏移量，从负极点转为正值（避开0值）
y_train_offset = y_train + output_offset  # 例：-600 → 600+60=660，-0.01 → 660-0.01=659.99
y_test_offset = y_test + output_offset    # 测试集用相同偏移量，保证一致性
print(f"偏移后训练集极点范围：{np.min(y_train_offset):.6f} ~ {np.max(y_train_offset):.6f}（均为正）")

# -------------------------- 数据标准化（输入+偏移后的输出） --------------------------
# 输入标准化（不变）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（用偏移后的正值，避免0值干扰）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_offset)  # 拟合偏移后的训练集
y_test_scaled = scaler_y.transform(y_test_offset)        # 测试集用训练集的标准化参数
print("数据标准化完成（输入直接标准化，输出偏移后标准化）")

# 转换为PyTorch张量并创建数据加载器（支持批量训练）
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)  # batch_size=128
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"数据加载器创建完成：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 定义神经网络模型（不变） --------------------------
class PolePredictor(nn.Module):
    def __init__(self):
        super(PolePredictor, self).__init__()
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

            # 输出层：32→3（保持与Excel一致的3个极点顺序）
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置（不变） --------------------------
# 初始化模型、损失函数、优化器
model = PolePredictor()
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
print(f"\n训练配置完成：总轮次{epochs}，初始学习率0.001，优化器Adam，输出偏移量{output_offset:.6f}")

# -------------------------- 5. 模型训练（含最佳模型保存，新增偏移量记录） --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印一次日志，测试损失下降时保存最佳模型）")
print("=" * 80)

for epoch in range(epochs):
    # -------------------------- 训练阶段 --------------------------
    model.train()  # 切换训练模式（启用Dropout/BatchNorm训练态）
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

    # -------------------------- 验证阶段（含最佳模型保存，新增偏移量记录） --------------------------
    model.eval()  # 切换评估模式（禁用Dropout/BatchNorm固定）
    with torch.no_grad():  # 禁用梯度计算，加速并节省内存
        outputs = model(X_test_tensor)
        test_loss = criterion(outputs, y_test_tensor).item()
        test_losses.append(test_loss)

        # 关键逻辑：仅当当前测试损失 < 历史最佳损失时，保存新最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss  # 更新最佳损失记录
            # 1. 保存模型参数（体积小，加载灵活）
            torch.save(model.state_dict(), 'best_pole_predictor.pth')
            # 2. 保存训练信息（新增output_offset，便于后续反标准化）
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'output_offset': output_offset,  # 记录偏移量，预测时需复用
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_y': {'mean': scaler_y.mean_.tolist(), 'std': scaler_y.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr']
            }
            np.save('best_model_info.npy', best_model_info)
            # 3. 打印保存日志
            print(f"Epoch {epoch + 1:4d}: 测试损失{test_loss:.6f}（历史最佳）→ 保存模型")

    # 学习率调度（根据测试损失调整）
    scheduler.step(test_loss)

    # 每10轮打印一次训练日志
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f} | 当前学习率: {current_lr:.6f}")

# -------------------------- 6. 加载最佳模型并评估（新增偏移量复用） --------------------------
print("\n" + "=" * 80)
print("训练结束，加载最佳模型进行评估")
print("=" * 80)

# 加载最佳模型（含异常处理，新增偏移量加载）
output_offset_loaded = output_offset  # 默认用当前计算的偏移量（防止加载失败）
try:
    # 加载模型参数
    model.load_state_dict(torch.load('best_pole_predictor.pth'))
    # 加载训练信息（含偏移量）
    best_info = np.load('best_model_info.npy', allow_pickle=True).item()
    output_offset_loaded = best_info['output_offset']  # 复用训练时的偏移量
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 对应训练轮次：Epoch {best_info['epoch']}")
    print(f"   - 最佳测试损失：{best_info['test_loss']:.6f}")
    print(f"   - 对应训练损失：{best_info['train_loss']:.6f}")
    print(f"   - 复用训练时的偏移量：{output_offset_loaded:.6f}")
    print(f"   - 标准化器参数已同步加载")
except FileNotFoundError:
    print("⚠️ 警告：未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}，使用当前计算的偏移量{output_offset_loaded:.6f}")
model.eval()  # 固定评估模式

# 在测试集上预测并反标准化（新增偏移量逆操作，恢复原始负极点）
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    # 反标准化步骤：1. 先反标准化到偏移后的正值 → 2. 减去偏移量恢复原始负极点
    y_pred_offset = scaler_y.inverse_transform(y_pred_scaled.numpy())  # 步骤1：反标准化到偏移尺度
    y_pred = y_pred_offset - output_offset_loaded  # 步骤2：减去偏移量，恢复原始负极点
    # 真实值反标准化（同样需偏移逆操作）
    y_true_offset = scaler_y.inverse_transform(y_test_scaled)
    y_true = y_true_offset - output_offset_loaded

# -------------------------- 7. 计算评估指标（修改：按Excel顺序标注极点） --------------------------
# 计算每个极点的平均绝对误差（MAE，按Excel顺序：Pole1→Pole2→Pole3）
mae_pole1 = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))  # 对应Excel的Pole1_Real
mae_pole2 = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))  # 对应Excel的Pole2_Real
mae_pole3 = np.mean(np.abs(y_pred[:, 2] - y_true[:, 2]))  # 对应Excel的Pole3_Real
total_mae = (mae_pole1 + mae_pole2 + mae_pole3) / 3  # 总平均MAE

print(f"\n📊 模型评估结果（平均绝对误差MAE）：")
print(f"   - 极点1（对应Excel Pole1_Real）：{mae_pole1:.4f}")
print(f"   - 极点2（对应Excel Pole2_Real）：{mae_pole2:.4f}")
print(f"   - 极点3（对应Excel Pole3_Real）：{mae_pole3:.4f}")
print(f"   - 总平均误差：{total_mae:.4f}")

# -------------------------- 8. 可视化结果（修改：按Excel顺序标注极点） --------------------------
# 8.1 绘制训练/测试损失曲线（不变）
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
plt.title('训练与测试损失曲线', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：loss_curve.png")

# 8.2 绘制前50个样本的预测值vs真实值对比（按Excel顺序标注极点）
sample_num = min(50, len(y_true))  # 取前50个样本（避免图太拥挤）
sample_indices = np.arange(sample_num)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

for i, ax in enumerate(axes):
    # 绘制真实值（蓝色实线）和预测值（红色虚线）
    ax.plot(sample_indices, y_true[sample_indices, i], color='#2E86AB', linewidth=2, label='真实值')
    ax.plot(sample_indices, y_pred[sample_indices, i], color='#FF0000', linewidth=1.5, linestyle='--', label='预测值')
    # 关键修改：按Excel列名标注极点，不做大小排序描述
    ax.set_title(f'极点{i + 1}（对应Excel Pole{i + 1}_Real，均为负）预测值与真实值对比（前{sample_num}个样本）', fontsize=12, pad=12)
    ax.set_ylabel('极点值（负值）', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[-1].set_xlabel('样本索引', fontsize=10)  # 最后一个子图加x轴标签
plt.tight_layout()
plt.savefig('pole_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 极点预测对比图已保存为：pole_prediction_comparison.png")

# -------------------------- 9. 打印前10条预测结果明细（按Excel顺序） --------------------------
print("\n" + "=" * 90)
print("前10条样本预测结果明细（与Excel顺序一致：Pole1→Pole2→Pole3，均为负）")
print("=" * 90)
print(
    f"{'样本':<6} {'真实Pole1':<12} {'预测Pole1':<12} {'真实Pole2':<12} {'预测Pole2':<12} {'真实Pole3':<12} {'预测Pole3':<12}")
print("-" * 90)
for i in range(min(10, len(y_pred))):
    print(
        f"{i:<6} {y_true[i, 0]:<12.4f} {y_pred[i, 0]:<12.4f} "
        f"{y_true[i, 1]:<12.4f} {y_pred[i, 1]:<12.4f} "
        f"{y_true[i, 2]:<12.4f} {y_pred[i, 2]:<12.4f}"
    )

print("\n" + "=" * 80)
print("所有文件已保存至当前目录：")
print("1. best_pole_predictor.pth → 最佳模型参数文件")
print("2. best_model_info.npy → 最佳模型训练信息（含偏移量+标准化参数）")
print("3. loss_curve.png → 训练/测试损失曲线")
print("4. pole_prediction_comparison.png → 极点预测对比图")
print("=" * 80)