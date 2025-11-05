import os
import pandas as pd
import re
import numpy as np


def txt_to_excel(txt_path, excel_save_dir):
    """只处理并保存实部和虚部数据（适配修改后的TXT格式，新增参数提取）"""
    txt_filename = os.path.basename(txt_path)
    excel_filename = txt_filename.replace(".txt", ".xlsx")
    excel_save_path = os.path.join(excel_save_dir, excel_filename)

    # 读取TXT文件
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        print(f"[处理中] {txt_filename}（共{len(lines)}行）")
    except Exception as e:
        print(f"[失败] {txt_filename}：读取错误 → {e}\n")
        return False

    # -------------------------- 新增：提取Cs/Rs/Rd参数（适配新TXT格式） --------------------------
    cs_val = None
    rs_val = None
    rd_val = None
    param_pattern = re.compile(r'【参数值】：Cs=(\d+\.?\d*)fF, Rs=(\d+\.?\d*)Ω, Rd=(\d+\.?\d*)Ω')  # 匹配新格式参数行
    for line in lines:
        line_strip = line.strip()
        param_match = param_pattern.match(line_strip)
        if param_match:
            cs_val = float(param_match.group(1))  # Cs值（如21fF → 21）
            rs_val = float(param_match.group(2))  # Rs值（如398Ω → 398）
            rd_val = float(param_match.group(3))  # Rd值（如247Ω → 247）
            break  # 找到参数行后退出循环
    # 校验参数是否提取成功
    if None in [cs_val, rs_val, rd_val]:
        print(f"[失败] {txt_filename}：未找到参数行（需包含'【参数值】：Cs=XXfF, Rs=XXΩ, Rd=XXΩ'）\n")
        return False
    print(f"  提取参数：Cs={cs_val}fF, Rs={rs_val}Ω, Rd={rd_val}Ω")

    # -------------------------- 数据段定位（保持逻辑，增强容错） --------------------------
    real_start_idx = None
    imag_start_idx = None
    db_start_idx = None
    for idx, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip == "【S21实部】：":
            real_start_idx = idx
        elif line_strip == "【S21虚部】：":
            imag_start_idx = idx
        elif line_strip == "【S21(dB)】：":
            db_start_idx = idx
    # 校验标记行完整性
    if None in [real_start_idx, imag_start_idx, db_start_idx]:
        print(f"[失败] {txt_filename}：未找到完整标记行（需同时包含'【S21实部】：'/'【S21虚部】：'/'【S21(dB)】：'）\n")
        return False

    # 正确划分数据段（跳过表头，避开dB数据）
    # 实部：从实部标记后2行（跳过"freq (Hz)         real(sp(2 1)) "表头）到虚部标记前1行
    real_data_lines = lines[real_start_idx + 2: imag_start_idx - 1]
    # 虚部：从虚部标记后2行（跳过表头）到dB标记前1行
    imag_data_lines = lines[imag_start_idx + 2: db_start_idx - 1]

    # 正则表达式：匹配频率（带G/M单位）和数值（带m/u/n单位，兼容正负值）
    data_pattern = re.compile(r'^\s*([\d\.]+[MG])\s{2,}([\d\.\-]+[munp]?)\s*$')

    # -------------------------- 提取频率和实部数据 --------------------------
    freq_list = []
    s21_real_list = []
    for line in real_data_lines:
        if not line.strip():
            continue
        match = data_pattern.match(line)
        if match:
            # 频率单位转换（G→1e9Hz，M→1e6Hz）
            freq_str = match.group(1)
            if "G" in freq_str:
                freq = float(freq_str.replace("G", "")) * 1e9
            else:  # "M"
                freq = float(freq_str.replace("M", "")) * 1e6
            freq_list.append(freq)

            # 实部数值单位转换（m→1e-3，u→1e-6，n→1e-9，无单位直接转浮点数）
            real_str = match.group(2)
            if "m" in real_str:
                real_val = float(real_str.replace("m", "")) * 1e-3
            elif "u" in real_str:
                real_val = float(real_str.replace("u", "")) * 1e-6
            elif "n" in real_str:
                real_val = float(real_str.replace("n", "")) * 1e-9
            else:
                real_val = float(real_str)
            s21_real_list.append(real_val)

    # -------------------------- 提取虚部数据 --------------------------
    s21_imag_list = []
    for line in imag_data_lines:
        if not line.strip():
            continue
        match = data_pattern.match(line)
        if match:
            # 虚部数值单位转换（同实部逻辑）
            imag_str = match.group(2)
            if "m" in imag_str:
                imag_val = float(imag_str.replace("m", "")) * 1e-3
            elif "u" in imag_str:
                imag_val = float(imag_str.replace("u", "")) * 1e-6
            elif "n" in imag_str:
                imag_val = float(imag_str.replace("n", "")) * 1e-9
            else:
                imag_val = float(imag_str)
            s21_imag_list.append(imag_val)

    # -------------------------- 数据长度校验 --------------------------
    data_len = len(freq_list)
    len_real = len(s21_real_list)
    len_imag = len(s21_imag_list)
    if data_len != len_real or data_len != len_imag:
        print(f"[失败] {txt_filename}：数据长度不匹配 → 频率：{data_len}，实部：{len_real}，虚部：{len_imag}\n")
        return False

    # -------------------------- 写入Excel（新增参数列） --------------------------
    df = pd.DataFrame({
        "频率(Hz)": freq_list,
        "频率(GHz)": [f / 1e9 for f in freq_list],  # 保留GHz便于查看
        "S21实部": s21_real_list,
        "S21虚部": s21_imag_list
    })

    try:
        with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='S21结果', index=False)
        print(f"[成功] {txt_filename} → 保存至：{excel_save_path}\n")
        return True
    except Exception as e:
        print(f"[失败] {txt_filename}：写入Excel错误 → {e}\n")
        return False


def batch_process_reim_txts(txt_source_dir="./reim", excel_target_dir="./reim_excel"):
    """批量处理主函数（逻辑不变，增强路径容错）"""
    # 检查源目录
    if not os.path.exists(txt_source_dir):
        print(f"错误：TXT源目录不存在 → {os.path.abspath(txt_source_dir)}")
        return
    # 创建目标目录
    if not os.path.exists(excel_target_dir):
        os.makedirs(excel_target_dir, exist_ok=True)  # 新增exist_ok=True，避免目录已存在时报错
        print(f"已创建Excel目录：{os.path.abspath(excel_target_dir)}\n")
    else:
        print(f"Excel目录：{os.path.abspath(excel_target_dir)}\n")

    # 筛选所有TXT文件
    txt_files = []
    for root, _, files in os.walk(txt_source_dir):
        for file in files:
            if file.lower().endswith(".txt") and "S21_Cs" in file:  # 新增文件名过滤，只处理S21参数文件
                txt_files.append(os.path.join(root, file))
    if not txt_files:
        print(f"未找到S21相关TXT文件 → {os.path.abspath(txt_source_dir)}")
        return

    # 批量处理
    print(f"===== 批量处理开始 =====")
    print(f"源目录：{os.path.abspath(txt_source_dir)}")
    print(f"待处理文件数：{len(txt_files)}\n")
    success_count = 0
    fail_count = 0
    fail_files = []
    for idx, txt_path in enumerate(txt_files, 1):
        txt_filename = os.path.basename(txt_path)
        print(f"[{idx}/{len(txt_files)}] 处理：{txt_filename}")
        if txt_to_excel(txt_path, excel_target_dir):
            success_count += 1
        else:
            fail_count += 1
            fail_files.append(txt_filename)

    # 处理结果总结
    print(f"===== 批量处理完成 =====")
    print(f"总文件数：{len(txt_files)}")
    print(f"成功数：{success_count}（保存至：{os.path.abspath(excel_target_dir)}）")
    print(f"失败数：{fail_count}")
    if fail_files:
        print(f"失败文件（{len(fail_files)}个）：{', '.join(fail_files[:10])}{'...' if len(fail_files) > 10 else ''}")
        print("建议：检查失败文件的「参数行格式」和「标记行完整性」")


if __name__ == "__main__":
    # 配置文件路径（根据实际情况修改）
    TXT_SOURCE_DIR = "C:\\Users\\zhaohongrui\\Desktop\\新建文件夹"  # 修改为你的TXT文件所在目录
    EXCEL_TARGET_DIR = "C:\\Users\\zhaohongrui\\Desktop\\新建文件夹"  # Excel保存目录
    # 执行批量处理
    batch_process_reim_txts(
        txt_source_dir=TXT_SOURCE_DIR,
        excel_target_dir=EXCEL_TARGET_DIR
    )