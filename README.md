# 向量拟合与极点预测工具包
一个用于S21参数处理、向量拟合及极点/留数预测的Python工具集，结合机器学习模型实现共轭极点判断与三极点预测功能。

## 项目简介
本项目提供了从原始S21参数数据处理到极点/留数预测的完整流程，主要功能包括：
- TXT格式S21数据批量转换为Excel
- 共轭极点分类判断（基于神经网络模型）
- 三极点参数预测（回归模型）
- 粒子群优化算法实现参数搜索

## 核心功能模块

### 1. 数据处理模块
- `txt脚本.py`：批量处理TXT格式的S21参数文件，提取实部、虚部数据及Cs/Rs/Rd参数，转换为Excel格式
- 支持频率单位（G/MHz）和数值单位（m/u/n）自动转换

### 2. 共轭极点分类
- `Conjugate.py`：定义共轭极点分类神经网络模型
- 输入：3个特征参数
- 输出：二分类结果（0:无共轭极点，1:有共轭极点）

### 3. 三极点预测
- `三极点预测.py`：实现三极点参数预测模型
- 基于多层神经网络的回归模型
- 支持数据标准化、自动偏移处理（针对负极点特性）
- 包含模型训练、评估及最佳模型保存功能


### 4. 优化搜索
- `search.py`：粒子群优化算法实现
- 用于参数空间搜索，寻找最优解

## 环境依赖
- Python 3.12+
- 主要依赖库：
  - pandas (数据处理)
  - numpy (数值计算)
  - torch (神经网络)
  - scikit-learn (数据预处理)
  - matplotlib (可视化)
  - openpyxl (Excel读写)

## 使用方法

### 数据处理
```python
# 批量处理TXT文件
from txt脚本 import batch_process_reim_txts

# 配置输入输出目录
TXT_SOURCE_DIR = "path/to/txt_files"
EXCEL_TARGET_DIR = "path/to/excel_files"

# 执行批量转换
batch_process_reim_txts(TXT_SOURCE_DIR, EXCEL_TARGET_DIR)
```

### 模型训练
```bash
# 三极点预测模型训练（位于数据分析目录）
python 3pole.py

# 或使用另一版本训练脚本
python 3pole_noconjugate/train3.py
````
### 参数优化
```python
# 粒子群优化搜索
from search import PSOOptimizer

# 初始化优化器并执行搜索
optimizer = PSOOptimizer(evaluator=your_evaluator)
best_pos, best_val, history = optimizer.optimize()

# 可视化优化结果
plot_results(evaluator, best_pos, history)
```
### 项目结构
```
向量拟合/
├── txt脚本.py           # TXT转Excel工具
├── Conjugate.py         # 共轭极点分类模型
├── search.py            # 粒子群优化算法
├── 三极点预测.py         # 三极点预测脚本
├── 拟合.py              # 向量拟合主程序
├── 对比图像.py          # 结果可视化工具
├── 数据分析/
│   └── 3pole.py         # 三极点预测模型训练
├── 3pole_noconjugate/
│   └── train3.py        # 保持极点顺序的训练脚本
└── reim_excel/          # 转换后的Excel文件输出目录
```
## 模型说明

1. **共轭极点分类模型**：
   - 输入：Cs、Rd、Rs三个器件参数
   - 网络结构：3→64→32→2（含ReLU、BatchNorm、Dropout）
   - 输出：二分类结果（是否为共轭极点场景）

2. **极点预测模型**：
   - 共轭场景：预测极点1实部/虚部、极点3实部（极点2由共轭关系推导）
   - 非共轭场景：直接预测三个实极点（保持原始顺序）
   - 均采用3→256→128→64→32→3的网络结构，支持自动偏移处理负极点

3. **留数预测模型**：
   - 共轭场景：3输入（器件参数）→3输出（留数），使用ReLU激活
   - 非共轭场景：9输入（3器件+3极点+3交叉特征）→3输出，使用LeakyReLU激活

## 许可证
具体请参见LICENSE文件

## 联系方式
如有问题或建议，请联系项目维护者。
- 邮箱：1668640479@qq.com
- B 站：祉佲






