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

# -------------------------- 2. 数据加载与预处理（已移除数据清洗） --------------------------
# 读取Excel数据（需确保文件路径正确）
try:
    excel_file = pd.ExcelFile('../S21批量拟合汇总结果(含直流项和比例项).xlsx')
    df = excel_file.parse('False')  # 读取非共轭极点的False工作表
    print(f"成功读取原始数据：共{len(df)}条样本，{len(df.columns)}列特征")
    # 校验必要列是否存在
    required_cols_base = ['Cs', 'Rd', 'Rs', 'Pole1_Real', 'Pole2_Real', 'Pole3_Real', 'Residue1_Real', 'Residue2_Real', 'Residue3_Real']
    missing_cols = [col for col in required_cols_base if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Excel缺少必要列：{', '.join(missing_cols)}")
except FileNotFoundError:
    raise FileNotFoundError("Excel文件未找到！请检查文件路径是否正确")
except Exception as e:
    raise Exception(f"读取数据失败：{str(e)}")

# -------------------------- 特征工程：9特征（6基础+3交叉） --------------------------
df_feat = df.copy()  # 直接使用原始数据构建特征，不做清洗

# 新增交叉特征
df_feat['Cs_Pole2'] = df_feat['Cs'] * df_feat['Pole2_Real']
df_feat['Pole2_Pole3'] = df_feat['Pole2_Real'] / (df_feat['Pole3_Real'] + 1e-6)  # 避免除零
df_feat['Cs_Pole3'] = df_feat['Cs'] * df_feat['Pole3_Real']

# 定义输入特征（9个）
input_feats = [
    'Cs', 'Rd', 'Rs', 'Pole1_Real', 'Pole2_Real', 'Pole3_Real',  # 6基础特征
    'Cs_Pole2', 'Pole2_Pole3', 'Cs_Pole3'  # 3交叉特征
]
print(f"\n特征工程完成：输入特征从6个扩展为9个（6基础+3交叉），使用原始数据（未清洗）")

# 提取输入和输出
X = df_feat[input_feats].values
y = df_feat[['Residue1_Real', 'Residue2_Real', 'Residue3_Real']].values
print(f"预测目标：3个原始留数（未执行异常值清洗）")

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, shuffle=True
)
print(f"\n数据集划分完成：训练集{len(X_train)}条，测试集{len(X_test)}条")
print(f"输入特征维度：{X_train.shape[1]}（9特征），输出留数维度：{y_train.shape[1]}（3留数）")

# 分析留数分布（原始数据，可能含极端值）
y_train_min = np.min(y_train)
y_train_max = np.max(y_train)
print(f"\n原始留数分布：训练集留数最小值={y_train_min:.6f}，最大值={y_train_max:.6f}（可能含极端值）")

# -------------------------- 数据标准化（9输入+原始留数） --------------------------
# 输入标准化（9特征）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 输出标准化（原始留数）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)
print("\n数据标准化完成：")
print(f" - 输入（9特征）：均值={scaler_X.mean_.round(4)[:3]}...，标准差={scaler_X.scale_.round(4)[:3]}...")
print(f" - 输出（原始留数）：均值={scaler_y.mean_.round(4)}，标准差={scaler_y.scale_.round(4)}")

# 创建数据加载器
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)
print(f"\n数据加载器创建完成：批量大小{train_loader.batch_size}，训练批次数{len(train_loader)}")


# -------------------------- 3. 神经网络模型（LeakyReLU激活） --------------------------
class ResiduePredictor(nn.Module):
    def __init__(self):
        super(ResiduePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 256),  # 9输入特征
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),  # 缓解死亡神经元
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),

            nn.Linear(32, 3)  # 3输出留数
        )

    def forward(self, x):
        return self.model(x)


# -------------------------- 4. 模型初始化与训练配置 --------------------------
model = ResiduePredictor()
criterion = nn.MSELoss()  # 使用MSE损失
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2正则化
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 训练超参数
epochs = 1000
best_test_loss = float('inf')
train_losses = []
test_losses = []
print(f"\n训练配置完成：")
print(f" - 总轮次：{epochs}，初始学习率：0.001，优化器：Adam（L2正则化1e-5）")
print(f" - 关键设置：9特征 + LeakyReLU激活（未执行异常值清洗）")

# -------------------------- 5. 模型训练 --------------------------
print("\n" + "=" * 80)
print("开始训练（每10轮打印日志，测试损失下降时保存最佳模型）")
print("=" * 80)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
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

        # 保存最佳模型（标记未清洗）
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_residue_predictor_9feat_leakyrelu.pth')
            best_model_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'input_feats': input_feats,
                'scaler_X': {'mean': scaler_X.mean_.tolist(), 'std': scaler_X.scale_.tolist()},
                'scaler_y': {'mean': scaler_y.mean_.tolist(), 'std': scaler_y.scale_.tolist()},
                'batch_size': train_loader.batch_size,
                'lr': optimizer.param_groups[0]['lr'],
                'cleaned': False  # 标记未执行异常值清洗
            }
            np.save('best_residue_info_9feat_leakyrelu.npy', best_model_info)
            print(f"Epoch {epoch + 1:4d}: 测试损失{test_loss:.6f}（历史最佳）→ 保存模型")

    # 学习率调度
    scheduler.step(test_loss)

    # 每10轮打印日志
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:4d}/{epochs}] | 训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f} | 学习率: {current_lr:.6f}")

# -------------------------- 6. 加载最佳模型并评估 --------------------------
print("\n" + "=" * 80)
print("训练结束，加载最佳模型评估（9特征+LeakyReLU，未清洗）")
print("=" * 80)

try:
    model.load_state_dict(torch.load('best_residue_predictor_9feat_leakyrelu.pth'))
    best_info = np.load('best_residue_info_9feat_leakyrelu.npy', allow_pickle=True).item()
    print(f"✅ 成功加载最佳模型：")
    print(f"   - 对应轮次：Epoch {best_info['epoch']}，最佳测试损失：{best_info['test_loss']:.6f}")
    print(f"   - 输入特征数：9（{best_info['input_feats']}），未执行异常值清洗")
except FileNotFoundError:
    print("⚠️ 警告：未找到最佳模型文件，使用最后一轮模型评估")
except Exception as e:
    print(f"❌ 加载模型失败：{str(e)}，使用当前模型评估")
model.eval()

# 预测与反标准化
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true = scaler_y.inverse_transform(y_test_scaled)

# -------------------------- 7. 计算评估指标 --------------------------
mae_res1 = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
mae_res2 = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
mae_res3 = np.mean(np.abs(y_pred[:, 2] - y_true[:, 2]))
total_mae = (mae_res1 + mae_res2 + mae_res3) / 3

print(f"\n📊 原始留数预测评估结果（平均绝对误差MAE）：")
print(f"   - 留数1：{mae_res1:.4f}")
print(f"   - 留数2：{mae_res2:.4f}")
print(f"   - 留数3：{mae_res3:.4f}")
print(f"   - 总平均误差：{total_mae:.4f}")

# -------------------------- 8. 可视化结果 --------------------------
# 8.1 损失曲线（标记未清洗）
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, color='#2E86AB', linewidth=1.5, label='训练损失')
plt.plot(range(1, epochs + 1), test_losses, color='#A23B72', linewidth=1.5, label='测试损失')
if 'best_info' in locals():
    best_epoch = best_info['epoch']
    best_loss = best_info['test_loss']
    plt.scatter(best_epoch, best_loss, color='red', s=50, zorder=5, label=f'最佳模型（Epoch{best_epoch}）')
plt.xlabel('训练轮次（Epoch）', fontsize=11)
plt.ylabel('损失值（MSE）', fontsize=11)
plt.title('9特征→3留数 训练与测试损失曲线（LeakyReLU，未清洗）', fontsize=13, pad=15)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('residue_loss_curve_9feat_leakyrelu.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\n📈 损失曲线已保存为：residue_loss_curve_9feat_leakyrelu.png")

# 8.2 留数预测对比图
sample_num = min(50, len(y_true))
sample_indices = np.arange(sample_num)
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(sample_indices, y_true[sample_indices, i], color='#2E86AB', linewidth=2, label='真实留数（原始）')
    ax.plot(sample_indices, y_pred[sample_indices, i], color='#FF0000', linewidth=1.5, linestyle='--', label='预测留数')
    ax.set_title(f'留数{i + 1}预测值与真实值对比（前{sample_num}个样本，未清洗）', fontsize=12, pad=12)
    ax.set_ylabel('留数值', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[-1].set_xlabel('样本索引', fontsize=10)
plt.tight_layout()
plt.savefig('residue_pred_comparison_9feat_leakyrelu.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"📊 留数对比图已保存为：residue_pred_comparison_9feat_leakyrelu.png")

# -------------------------- 9. 打印前10条预测结果明细 --------------------------
print("\n" + "=" * 110)
print("前10条样本预测结果明细（9特征+LeakyReLU，未清洗）")
print("=" * 110)
print(
    f"{'样本':<6} {'真实Res1':<12} {'预测Res1':<12} {'真实Res2':<12} {'预测Res2':<12} {'真实Res3':<12} {'预测Res3':<12}")
print("-" * 110)
for i in range(min(10, len(y_pred))):
    print(
        f"{i:<6} {y_true[i, 0]:<12.4f} {y_pred[i, 0]:<12.4f} "
        f"{y_true[i, 1]:<12.4f} {y_pred[i, 1]:<12.4f} "
        f"{y_true[i, 2]:<12.4f} {y_pred[i, 2]:<12.4f}"
    )

# -------------------------- 10. 输出文件汇总 --------------------------
print("\n" + "=" * 80)
print("所有文件已保存至当前目录（9特征+LeakyReLU，未清洗）：")
print("1. best_residue_predictor_9feat_leakyrelu.pth → 最佳模型参数")
print("2. best_residue_info_9feat_leakyrelu.npy → 训练信息（含未清洗标记）")
print("3. residue_loss_curve_9feat_leakyrelu.png → 损失曲线")
print("4. residue_pred_comparison_9feat_leakyrelu.png → 留数对比图")
print("=" * 80)