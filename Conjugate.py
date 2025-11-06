import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 新增：用于保存标准化器

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 读取数据（请替换为你的文件路径）
excel_file = pd.ExcelFile('Data/S21批量拟合汇总结果(含直流项和比例项).xlsx')
df = excel_file.parse('All')

# 查看Conjugate?列的分布
print("Conjugate?列数据分布：")
print(df['Conjugate?'].value_counts())
print("-" * 50)

# 提取输入特征（前三列器件参数）和目标值（Conjugate?列）
X = df.iloc[:, :3].values  # 前三列作为输入
y = df['Conjugate?'].values  # 目标列（True/False）

# 将True/False转换为1/0（分类任务需要数值标签）
y = np.where(y, 1, 0)

# 划分训练集和测试集（8:2比例，保持类别平衡）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合标准化器
X_test_scaled = scaler.transform(X_test)        # 测试集复用训练集的标准化参数

# 新增：保存标准化器为.pkl文件（供预测阶段使用）
scaler_path = "classifier_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"标准化器已保存至：{scaler_path}")
print("-" * 50)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)


# 定义分类神经网络模型
class ConjugateClassifier(nn.Module):
    def __init__(self):
        super(ConjugateClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2个输出（0:无共轭极点，1:有共轭极点）
        )

    def forward(self, x):
        return self.model(x)


# 初始化模型、损失函数和优化器
model = ConjugateClassifier()
criterion = nn.CrossEntropyLoss()  # 分类任务专用损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 1500
batch_size = 32
best_accuracy = 0.0
best_model_path = "best_conjugate_classifier.pth"  # 最佳模型保存路径

for epoch in range(epochs):
    # 训练模式
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    total_train = 0
    correct_train = 0
    train_loss = 0.0

    for i in range(0, len(X_train_tensor), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # 计算训练集指标
        train_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()

    # 验证模式
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        val_loss = criterion(outputs, y_test_tensor).item()
        _, predicted = torch.max(outputs.data, 1)
        val_accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        # 保存最佳模型（验证准确率最高的模型）
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"第{epoch+1}轮更新最佳模型，验证准确率: {val_accuracy:.4f}")

    # 每10轮打印一次训练进度
    if (epoch + 1) % 10 == 0:
        train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"训练损失: {train_loss / len(X_train_tensor):.4f}, 训练准确率: {train_accuracy:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
        print("-" * 50)

# 加载最佳模型进行最终评估
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs.data, 1)

    # 计算准确率
    test_accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f"\n最终测试集准确率: {test_accuracy:.4f}")  # 打印核心准确率

    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(
        y_test_tensor.numpy(),
        y_pred.numpy(),
        target_names=['无共轭极点 (False)', '有共轭极点 (True)']
    ))

    # 绘制并保存混淆矩阵图片
    cm = confusion_matrix(y_test_tensor.numpy(), y_pred.numpy())
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['无共轭极点', '有共轭极点'],
                yticklabels=['无共轭极点', '有共轭极点'])
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    plt.title('共轭极点分类混淆矩阵')
    confusion_matrix_path = "conjugate_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵已保存为: {confusion_matrix_path}")

    # 打印混淆矩阵数值
    print("\n混淆矩阵数值:")
    print(cm)
    print("\n矩阵说明:")
    print("行: 实际类别 (0=无, 1=有)")
    print("列: 预测类别 (0=无, 1=有)")

# 打印部分预测结果（前10条）
print("\n前10条预测结果对比:")
print(f"{'实际值':<10} {'预测值':<10} {'是否正确'}")
print("-" * 30)
for i in range(min(10, len(y_pred))):
    actual = "True" if y_test[i] == 1 else "False"
    pred = "True" if y_pred[i] == 1 else "False"
    correct = "✓" if actual == pred else "✗"
    print(f"{actual:<10} {pred:<10} {correct}")