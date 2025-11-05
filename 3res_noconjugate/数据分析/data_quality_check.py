import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.stats import pearsonr  # 计算皮尔逊相关系数及显著性
import seaborn as sns  # 绘制美观的相关性热力图
import warnings
warnings.filterwarnings('ignore')  # 屏蔽无关警告


# -------------------------- 1. 基础配置：设置中文字体+结果保存目录 --------------------------
def set_chinese_font():
    """设置中文字体，避免绘图中文乱码"""
    try:
        fm.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    except:
        try:
            fm.fontManager.addfont('/Library/Fonts/Songti.ttc')
            plt.rcParams['font.sans-serif'] = ['Songti SC', 'DejaVu Sans']
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def create_save_dir(save_dir='./correlation_analysis'):
    """创建结果保存目录，避免路径不存在报错"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# -------------------------- 2. 核心函数：数据加载与校验 --------------------------
def load_and_validate_data(excel_path):
    """
    加载Excel数据，校验9个必要列是否存在，返回用于相关性分析的DataFrame
    :param excel_path: Excel文件路径
    :return: corr_df: 包含9个变量的DataFrame（列名简化为短名称）
    """
    # 读取Excel
    try:
        excel_file = pd.ExcelFile(excel_path)
        df = excel_file.parse('False')  # 读取非共轭极点的"False"工作表
        print(f"✅ 成功读取Excel：共{len(df)}条样本，{len(df.columns)}列特征")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Excel文件未找到！请检查路径：{excel_path}")
    except Exception as e:
        raise Exception(f"❌ 读取Excel失败：{str(e)}")

    # 校验9个必要列是否存在
    required_cols = [
        'Cs', 'Rd', 'Rs',          # 3个器件参数
        'Pole1_Real', 'Pole2_Real', 'Pole3_Real',  # 3个实极点
        'Residue1_Real', 'Residue2_Real', 'Residue3_Real'  # 3个留数
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Excel缺少必要列：{', '.join(missing_cols)}")

    # 提取9个变量，简化列名（便于绘图标注）
    corr_df = df[required_cols].copy()
    corr_df.columns = ['Cs', 'Rd', 'Rs', 'Pole1', 'Pole2', 'Pole3', 'Res1', 'Res2', 'Res3']

    # 简单数据清洗：删除含NaN的行（避免影响相关性计算）
    initial_count = len(corr_df)
    corr_df = corr_df.dropna()
    if len(corr_df) < initial_count:
        print(f"⚠️ 检测到{initial_count - len(corr_df)}条含NaN的样本，已自动删除，剩余{len(corr_df)}条有效样本")

    print(f"✅ 数据准备完成：9个变量（6输入+3输出）→ {corr_df.columns.tolist()}")
    return corr_df


# -------------------------- 3. 核心函数：计算相关系数与显著性 --------------------------
def calculate_correlation(corr_df):
    """
    计算9个变量的皮尔逊相关系数矩阵和显著性p值矩阵
    :param corr_df: 包含9个变量的DataFrame
    :return: corr_matrix: 相关系数矩阵（DataFrame）, p_matrix: 显著性p值矩阵（DataFrame）
    """
    print(f"\n📊 开始计算相关系数与显著性（共{len(corr_df.columns)}个变量）...")
    var_names = corr_df.columns.tolist()
    n_vars = len(var_names)

    # 初始化相关系数矩阵和p值矩阵
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))

    # 遍历所有变量对，计算相关系数和p值
    for i in range(n_vars):
        for j in range(n_vars):
            var1 = corr_df[var_names[i]]
            var2 = corr_df[var_names[j]]
            # 计算皮尔逊相关系数（corr）和显著性（p_val）
            corr, p_val = pearsonr(var1, var2)
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_val

    # 转换为DataFrame（便于后续保存和标注）
    corr_matrix = pd.DataFrame(corr_matrix, index=var_names, columns=var_names).round(4)
    p_matrix = pd.DataFrame(p_matrix, index=var_names, columns=var_names).round(4)

    print(f"✅ 相关系数矩阵计算完成（范围：-1~1）")
    print(f"✅ 显著性p值矩阵计算完成（p<0.05为显著相关）")
    return corr_matrix, p_matrix


# -------------------------- 4. 核心函数：绘制相关性可视化图表 --------------------------
def plot_correlation_heatmap(corr_matrix, p_matrix, save_dir):
    """
    绘制相关性热力图（标注相关系数，显著相关用粗体突出）
    :param corr_matrix: 相关系数矩阵（DataFrame）
    :param p_matrix: 显著性p值矩阵（DataFrame）
    :param save_dir: 图表保存目录
    """
    plt.figure(figsize=(12, 10))
    # 绘制热力图：用RdBu_r色卡（蓝→白→红，对应负相关→无相关→正相关）
    im = plt.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)

    # 添加颜色条（标注相关系数范围）
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('皮尔逊相关系数', fontsize=12, labelpad=10)

    # 标注相关系数（显著相关的系数用粗体）
    var_names = corr_matrix.columns.tolist()
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            # 显著性判断：p<0.05为显著，标注为粗体；否则正常字体
            font_weight = 'bold' if (p_val < 0.05 and i != j) else 'normal'
            # 文字颜色：系数绝对值>0.5用白色（避免与背景色冲突）
            font_color = 'white' if abs(corr_val) > 0.5 else 'black'
            # 标注相关系数（保留3位小数）
            plt.text(j, i, f'{corr_val:.3f}',
                     ha='center', va='center', fontsize=9,
                     fontweight=font_weight, color=font_color)

    # 设置坐标轴标签
    plt.xticks(range(len(var_names)), var_names, rotation=45, ha='right', fontsize=11)
    plt.yticks(range(len(var_names)), var_names, fontsize=11)
    # 设置标题（注明显著相关的标注规则）
    plt.title('9变量相关性热力图（6输入特征+3留数目标）\n注：粗体表示p<0.05的显著相关',
              fontsize=14, pad=20)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 相关性热力图已保存：{save_path}")


def plot_input_output_scatter(corr_df, save_dir):
    """
    绘制输入特征与留数目标的散点图矩阵（重点展示6输入→3输出的关系）
    :param corr_df: 包含9个变量的DataFrame
    :param save_dir: 图表保存目录
    """
    # 拆分输入变量和输出变量
    input_vars = ['Cs', 'Rd', 'Rs', 'Pole1', 'Pole2', 'Pole3']  # 6个输入
    output_vars = ['Res1', 'Res2', 'Res3']  # 3个输出

    # 创建3行6列的子图布局
    fig, axes = plt.subplots(nrows=len(output_vars), ncols=len(input_vars), figsize=(18, 12))
    fig.suptitle('输入特征与留数目标散点图矩阵（每个点代表1个样本）', fontsize=15, y=0.98)

    # 遍历每个留数（行）和每个输入特征（列），绘制散点图
    for row_idx, res_var in enumerate(output_vars):
        for col_idx, input_var in enumerate(input_vars):
            ax = axes[row_idx, col_idx]
            # 绘制散点图（透明度0.6，避免点重叠；点大小20，平衡清晰度和密度）
            ax.scatter(corr_df[input_var], corr_df[res_var],
                      alpha=0.6, s=20, color='#2E86AB', edgecolor='none')

            # 计算当前输入-输出对的相关系数和显著性
            corr_val, p_val = pearsonr(corr_df[input_var], corr_df[res_var])
            # 显著性标注：p<0.05用"**"，否则无标注
            sig_mark = '**' if p_val < 0.05 else ''
            # 设置子图标题（标注变量对、相关系数、显著性）
            ax.set_title(f'{input_var} vs {res_var}\ncorr={corr_val:.3f}{sig_mark}',
                        fontsize=10, pad=8)

            # 设置坐标轴标签（字体大小9，避免拥挤）
            ax.set_xlabel(input_var, fontsize=9)
            ax.set_ylabel(res_var, fontsize=9)
            # 添加网格（便于观察数据趋势）
            ax.grid(True, linestyle='--', alpha=0.3)

    # 调整子图间距，避免标题和标签被遮挡
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    save_path = os.path.join(save_dir, 'input_output_scatter_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 输入-输出散点图矩阵已保存：{save_path}")


# -------------------------- 5. 核心函数：保存结果+输出分析总结 --------------------------
def save_correlation_results(corr_matrix, p_matrix, save_dir):
    """
    将相关系数矩阵和p值矩阵保存到Excel文件
    :param corr_matrix: 相关系数矩阵（DataFrame）
    :param p_matrix: 显著性p值矩阵（DataFrame）
    :param save_dir: 保存目录
    """
    excel_path = os.path.join(save_dir, 'correlation_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        corr_matrix.to_excel(writer, sheet_name='相关系数矩阵', index=True)
        p_matrix.to_excel(writer, sheet_name='显著性p值矩阵', index=True)
    print(f"✅ 量化结果已保存至Excel：{excel_path}")


def print_correlation_summary(corr_matrix, p_matrix):
    """
    输出相关性分析总结（强相关变量、输入-输出显著相关）
    :param corr_matrix: 相关系数矩阵（DataFrame）
    :param p_matrix: 显著性p值矩阵（DataFrame）
    """
    var_names = corr_matrix.columns.tolist()
    input_vars = ['Cs', 'Rd', 'Rs', 'Pole1', 'Pole2', 'Pole3']
    output_vars = ['Res1', 'Res2', 'Res3']

    print(f"\n" + "="*80)
    print("📋 相关性分析总结")
    print("="*80)

    # 1. 输出强相关变量（|corr| > 0.7且p<0.05）
    print(f"\n1. 强相关变量（|corr| > 0.7 且 p<0.05）：")
    strong_corr = []
    for i in range(len(var_names)):
        for j in range(i+1, len(var_names)):  # 避免重复计算（i<j）
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_matrix.iloc[i, j]
            if abs(corr_val) > 0.7 and p_val < 0.05:
                var1 = var_names[i]
                var2 = var_names[j]
                strong_corr.append(f"   - {var1} ↔ {var2}（corr={corr_val:.3f}）")
    if strong_corr:
        for item in strong_corr:
            print(item)
    else:
        print(f"   - 无|corr|>0.7的显著强相关变量")

    # 2. 输出输入-输出显著相关（p<0.05）
    print(f"\n2. 输入-输出显著相关（p<0.05）：")
    input_output_corr = []
    for input_var in input_vars:
        for output_var in output_vars:
            corr_val = corr_matrix.loc[input_var, output_var]
            p_val = p_matrix.loc[input_var, output_var]
            if p_val < 0.05:
                input_output_corr.append(f"   - {input_var} → {output_var}（corr={corr_val:.3f}）")
    if input_output_corr:
        for item in input_output_corr:
            print(item)
    else:
        print(f"   - 无输入与输出的显著相关变量")

    print(f"\n" + "="*80)


# -------------------------- 6. 主函数：串联所有流程 --------------------------
def main(excel_path='../../S21批量拟合汇总结果(含直流项和比例项).xlsx'):
    """
    主函数：串联数据加载→相关性计算→可视化→结果保存→总结输出
    :param excel_path: Excel文件路径（可根据实际情况修改）
    """
    # 1. 基础配置
    set_chinese_font()
    save_dir = create_save_dir()
    print(f"📌 结果将保存至：{os.path.abspath(save_dir)}")

    # 2. 数据加载与校验
    corr_df = load_and_validate_data(excel_path)

    # 3. 计算相关系数和显著性
    corr_matrix, p_matrix = calculate_correlation(corr_df)

    # 4. 绘制可视化图表
    plot_correlation_heatmap(corr_matrix, p_matrix, save_dir)
    plot_input_output_scatter(corr_df, save_dir)

    # 5. 保存量化结果
    save_correlation_results(corr_matrix, p_matrix, save_dir)

    # 6. 输出分析总结
    print_correlation_summary(corr_matrix, p_matrix)

    print(f"\n🎉 所有相关性分析任务完成！结果已保存至：{os.path.abspath(save_dir)}")


# -------------------------- 7. 执行分析（修改Excel路径后运行） --------------------------
if __name__ == "__main__":
    # ！！！重要：根据你的Excel文件实际路径修改以下参数！！！
    EXCEL_FILE_PATH = '../../S21批量拟合汇总结果(含直流项和比例项).xlsx'  # 你的Excel文件路径
    main(excel_path=EXCEL_FILE_PATH)