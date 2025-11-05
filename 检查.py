import os

import pandas as pd


def check_excel_nulls(directory='./reim_excel'):
    """
    检查指定目录下所有Excel文件的空值，精确到具体单元格位置
    :param directory: 要检查的Excel目录，默认./reim_excel
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"❌ 错误：目录不存在 → {os.path.abspath(directory)}")
        return

    # 获取所有.xlsx文件
    excel_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.xlsx') and not file.startswith('~$'):  # 排除临时文件
            excel_files.append(os.path.join(directory, file))

    if not excel_files:
        print(f"ℹ️ 提示：目录 {os.path.abspath(directory)} 中未找到任何Excel文件")
        return

    print(f"===== 开始空值检查 =====")
    print(f"检查目录：{os.path.abspath(directory)}")
    print(f"待检查文件总数：{len(excel_files)}\n")

    # 遍历每个文件检查空值
    for file_idx, file_path in enumerate(excel_files, 1):
        file_name = os.path.basename(file_path)
        print(f"[{file_idx}/{len(excel_files)}] 检查文件：{file_name}")

        try:
            # 读取Excel文件（只读取第一个工作表）
            df = pd.read_excel(file_path, sheet_name=0)
            total_cells = df.size  # 总单元格数

            # 检测所有空值位置（返回布尔值DataFrame）
            null_mask = df.isnull()

            # 统计空值总数
            null_count = null_mask.sum().sum()

            if null_count == 0:
                print(f"✅ 无空值（总单元格：{total_cells}）\n")
                continue

            # 存在空值：列出具体位置
            print(f"❌ 发现空值：共{null_count}个空值（总单元格：{total_cells}）")

            # 遍历每一列查找空值
            for col in df.columns:
                # 获取该列中空值所在的行索引（注意：Excel行号从1开始，索引从0开始）
                null_rows = null_mask[col][null_mask[col] == True].index.tolist()
                if null_rows:
                    # 转换为Excel实际行号（索引+2，因为表头占1行）
                    excel_rows = [str(row + 2) for row in null_rows]
                    print(f"  列名：{col}，空值行号：{', '.join(excel_rows)}")

            print()  # 空行分隔

        except Exception as e:
            print(f"⚠️ 读取文件失败：{str(e)}\n")

    print("===== 检查完成 =====")
    print(f"总检查文件数：{len(excel_files)}")


if __name__ == "__main__":
    # 可修改为实际Excel目录路径
    EXCEL_DIRECTORY = "./reim_excel"
    check_excel_nulls(directory=EXCEL_DIRECTORY)
