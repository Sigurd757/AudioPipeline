#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 06: 最终统计与文件归档

本脚本是音频处理流程的最后一步，承接 `05_classification_full.py` 的输出。

核心目标：
1.  读取步骤 05 的完整 SCP (其中包含步骤 04 的 VAD 结果)。
2.  根据 VAD 结果进行 "三分类"，将音频分为：
    -   `Clean` (纯净): 0 个 VAD 模型检测到人声。
    -   `Suspected` (疑似): 1 个 VAD 模型检测到人声。
    -   `Rejected` (拒绝): >1 个 VAD 模型检测到人声。
3.  为这三类分别生成独立的 SCP 文件。
4.  对这三类音频，分别按步骤 05 的 `fused_label` (声音类别) 和 `dataset` 
    (数据集) 进行详细的数量和时长统计。
5.  生成一份完整的 `stats.json` 和 `stats.txt` 报告。
6.  (可选) 将这三类音频文件按 `数据集/声音类别/` 的结构复制到指定的
    不同归档目录中。
"""

import argparse
import logging
import os
import pandas as pd
import shutil
from tqdm import tqdm
import json
import csv

def setup_logger(log_dir):
    """
    配置并返回一个日志记录器 (Logger)。
    (此函数与之前脚本中的功能相同)
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("filter_and_sort") # 日志记录器名称
    logger.setLevel(logging.INFO)
    logger.propagate = False # 防止日志向上传播
    if logger.handlers: logger.handlers.clear() # 清除已有的处理器
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 日志文件名
    file_handler = logging.FileHandler(os.path.join(log_dir, "06_filter_and_sort.log"), encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# ==============================================================================
# 核心修改 1：创建一个可重用的统计函数
# ==============================================================================
def analyze_and_log_stats(dataframe, category_name, logger):
    """
    对传入的 DataFrame 按 'fused_label' 进行分组统计，并记录日志。
    
    参数:
        dataframe (pd.DataFrame): 待分析的数据 (例如 clean_df)。
        category_name (str): 类别的名称 (例如 "Clean")，用于日志。
        logger (logging.Logger): 日志记录器实例。
        
    返回:
        tuple: (stats_dict, summary)
            - stats_dict (list): 详细统计结果的字典列表。
            - summary (dict): 总体统计结果的字典。
    """
    logger.info(f"--- 正在统计 '{category_name}' 类别 ---")
    if dataframe.empty or 'fused_label' not in dataframe.columns:
        logger.info(f"'{category_name}' 类别为空，或缺少 'fused_label' 列，跳过统计。")
        return [], {'total_files': 0, 'total_duration_seconds': 0, 'total_duration_hms': "00:00:00.00"}
        
    # 确保 duration 列为数值类型
    dataframe['duration'] = pd.to_numeric(dataframe['duration'], errors='coerce').fillna(0)
    
    # 从 'original_audio_id' (例如 "DatasetA_file001_chunk001") 提取数据集名称
    if 'dataset' not in dataframe.columns:
         dataframe['dataset'] = dataframe['original_audio_id'].apply(lambda x: x.split('_')[0])
    
    # 核心统计：按 [数据集] 和 [声音类别] 两个维度进行分组
    stats_by_class = dataframe.groupby(['dataset', 'fused_label']).agg(
        count=('fused_label', 'size'),          # 统计文件数
        total_duration_seconds=('duration', 'sum') # 统计总时长
    ).sort_values(by=['dataset', 'count'], ascending=[True, False]).reset_index()

    # 计算总计
    total_count = stats_by_class['count'].sum()
    total_duration_seconds = stats_by_class['total_duration_seconds'].sum()
    # 格式化时长为 HH:MM:SS.ss
    total_duration_hms = f"{int(total_duration_seconds/3600):02d}:{int((total_duration_seconds%3600)/60):02d}:{total_duration_seconds%60:05.2f}"
    
    stats_dict = stats_by_class.to_dict('records')
    summary = {'total_files': int(total_count), 'total_duration_seconds': round(total_duration_seconds, 4), 'total_duration_hms': total_duration_hms}
    
    # 打印详细日志
    for record in stats_dict:
        logger.info(f"  > 数据集: {record['dataset']:<15} | 类别: {record['fused_label']:<25} | 文件数: {record['count']}")
    logger.info(f"  > '{category_name}' 类别总计: {summary['total_files']} 个文件, 总时长: {summary['total_duration_hms']}")
    
    return stats_dict, summary

def run_screener_stats_and_copy(args, logger):
    """
    执行筛选、统计和复制的主函数。
    """
    logger.info(f"读取VAD和分类的完整清单: {args.input_scp}")
    try:
        # 使用 pandas 读取步骤05的输出
        # 步骤05使用 `quoting=csv.QUOTE_NONE` 写入，
        # 但步骤04写入时，'vad_results_json' 列可能包含转义的引号。
        # pandas 的默认 C 解析器 (engine='c') 通常能正确处理标准转义。
        df = pd.read_csv(
            args.input_scp, 
            sep='\t', 
            header=0, 
            dtype=str, 
            keep_default_na=False, 
            encoding='utf-8'
        )
    except pd.errors.EmptyDataError:
        logger.warning(f"输入文件 {args.input_scp} 为空。"); return
    except FileNotFoundError:
        logger.error(f"输入文件未找到: {args.input_scp}")
        return

    logger.info(f"开始根据VAD结果进行三分类筛选...")
    logger.info(f"使用的VAD模型: {args.vad_models}")
    logger.info(f"语音比例阈值: {args.speech_ratio_threshold}")

    fail_counts, categories = [], []
    
    # -------------------------------------
    # 1. VAD 三分类逻辑
    # -------------------------------------
    for index, row in tqdm(df.iterrows(), total=len(df), desc="分析VAD结果进行三分类"):
        try:
            # 解析 'vad_results_json' 列
            vad_scores = json.loads(row['vad_results_json'])
            
            # 计算 "失败" (即检测到人声) 的 VAD 模型数量
            fail_count = sum(
                1 for model in args.vad_models 
                if float(vad_scores.get(model, "0.0")) > args.speech_ratio_threshold
            )
            fail_counts.append(fail_count)
            
            # 根据失败数量分配类别
            if fail_count == 0: 
                categories.append("Clean")     # 0 个模型检测到人声 -> 纯净
            elif fail_count == 1: 
                categories.append("Suspected") # 1 个模型检测到人声 -> 疑似
            else: 
                categories.append("Rejected")  # >1 个模型检测到人声 -> 拒绝
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"处理行 {index} (chunk_id: {row['chunk_id']}) 出错: {e}. 归为 'Rejected'.")
            fail_counts.append(len(args.vad_models)); categories.append("Rejected")
            
    df['fail_count'] = fail_counts
    df['category'] = categories

    # -------------------------------------
    # 2. 拆分 DataFrame 并保存 SCP
    # -------------------------------------
    clean_df = df[df['category'] == 'Clean'].copy()
    suspected_df = df[df['category'] == 'Suspected'].copy()
    rejected_df = df[df['category'] == 'Rejected'].copy()

    logger.info("-" * 30)
    logger.info(f"VAD分类统计: Clean={len(clean_df)}, Suspected={len(suspected_df)}, Rejected={len(rejected_df)}")
    logger.info("-" * 30)

    # 使用 Pandas 的 to_csv 写入。
    # Pandas 默认会使用 rfc4180 标准（即 csv.QUOTE_MINIMAL），
    # 它会自动为包含分隔符或引号的字段（如JSON列）添加双引号。
    os.makedirs(os.path.dirname(args.output_clean_scp), exist_ok=True)
    clean_df.to_csv(args.output_clean_scp, sep='\t', index=False, encoding='utf-8')
    suspected_df.to_csv(args.output_suspected_scp, sep='\t', index=False, encoding='utf-8')
    rejected_df.to_csv(args.output_rejected_scp, sep='\t', index=False, encoding='utf-8')
    logger.info(f"分类清单已保存。")
    
    # ==============================================================================
    # 核心修改 2：对所有三个类别进行统计
    # ==============================================================================
    logger.info("开始对所有VAD类别进行详细的声音分类统计...")
    
    # 3. 分别调用统计函数
    clean_stats, clean_summary = analyze_and_log_stats(clean_df, "Clean", logger)
    suspected_stats, suspected_summary = analyze_and_log_stats(suspected_df, "Suspected", logger)
    rejected_stats, rejected_summary = analyze_and_log_stats(rejected_df, "Rejected", logger)

    # 4. 汇总所有统计结果
    final_stats_data = {
        "clean_noise_stats": {
            "by_dataset_and_class": clean_stats,
            "summary": clean_summary
        },
        "suspected_noise_stats": {
            "by_dataset_and_class": suspected_stats,
            "summary": suspected_summary
        },
        "rejected_noise_stats": {
            "by_dataset_and_class": rejected_stats,
            "summary": rejected_summary
        }
    }
    
    # 5. 写入 JSON 报告
    with open(args.stats_json_path, 'w', encoding='utf-8') as f_json:
        json.dump(final_stats_data, f_json, indent=4, ensure_ascii=False)

    # 6. 写入 TXT 报告 (人类可读)
    with open(args.stats_txt_path, 'w', encoding='utf-8') as f_txt:
        # 定义一个辅助函数来写入报告
        def write_report_section(title, stats, summary):
            f_txt.write(f"--- {title} ---\n")
            if not stats:
                f_txt.write("N/A (无文件)\n\n")
                return
            # 写入详细条目
            for record in stats:
                f_txt.write(f"  Dataset: {record['dataset']:<15} | Class: {record['fused_label']:<25} | Files: {record['count']}\n")
            f_txt.write("-" * 60 + "\n")
            # 写入总结
            f_txt.write(f"  Total:   {'':<15} | {'':<25} | Files: {summary['total_files']} (Duration: {summary['total_duration_hms']})\n\n")

        write_report_section("Statistics of Final Clean Noise (VAD=Clean)", clean_stats, clean_summary)
        write_report_section("Statistics of Suspected Noise (VAD=Suspected)", suspected_stats, suspected_summary)
        write_report_section("Statistics of Rejected Noise (VAD=Rejected)", rejected_stats, rejected_summary)
        
    logger.info(f"完整的分类统计报告已保存至: {args.stats_json_path} 和 {args.stats_txt_path}")

    # -------------------------------------
    # 7. (可选) 文件复制/归档
    # -------------------------------------
    def copy_files(dataframe, dest_dir, source_root_dir, desc):
        """
        复制文件到目标目录，并按 `数据集/声音类别/` 组织。
        
        参数:
            dataframe (pd.DataFrame): 包含要复制文件信息的 DF (需要 'final_path', 'original_audio_id', 'fused_label' 列)。
            dest_dir (str | None): 目标根目录。
            source_root_dir (str | None): 源文件的根目录 (用于计算相对路径)。
            desc (str): 类别描述 (用于日志)。
        """
        # 如果目标目录或源根目录未提供，或者 DF 为空，则跳过
        if not dest_dir or not source_root_dir or dataframe.empty:
            if dest_dir and dataframe.empty: logger.info(f"'{desc}' 类别为空，无需复制。")
            return
            
        logger.info(f"开始复制 {len(dataframe)} 个 '{desc}' 文件..."); logger.info(f"来源根目录: {source_root_dir}"); logger.info(f"目标目录: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)
        
        # 确保 'dataset' 和 'fused_label' 列存在
        if 'dataset' not in dataframe.columns: dataframe['dataset'] = dataframe['original_audio_id'].apply(lambda x: x.split('_')[0])
        if 'fused_label' not in dataframe.columns: dataframe['fused_label'] = 'Unknown_Class'
        
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"复制 {desc} 文件"):
            source_path = row["final_path"] # 这是文件的绝对路径
            try:
                relative_path = os.path.relpath(source_path, source_root_dir)
                audio_class = row["fused_label"].replace('/', '_').replace(' ', '_')
                if not audio_class: audio_class = "Unknown_Class"
                
                path_parts = relative_path.split(os.sep)
                if len(path_parts) > 1:
                    dataset_name = path_parts[1]
                    original_sub_path_parts = path_parts[2:]
                    original_sub_path = os.path.join(*original_sub_path_parts) if original_sub_path_parts else os.path.basename(source_path)
                    
                else: 
                    # 路径解析失败的备用方案
                    dataset_name = "Unknown_Dataset"
                    original_sub_path = os.path.basename(source_path)

                # 4. 构建目标路径：Dest_Dir / Dataset / Class / file.wav
                destination_path = os.path.join(dest_dir, dataset_name, audio_class, original_sub_path)
                
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy2(source_path, destination_path) # copy2 保留元数据
            except Exception as e:
                logger.error(f"复制文件失败: {source_path} -> {destination_path}, 错误: {e}")
    
    # 根据参数调用复制函数
    if args.source_root_dir:
        copy_files(clean_df, args.copy_clean_to, args.source_root_dir, "纯净(Clean)")
        copy_files(suspected_df, args.copy_suspected_to, args.source_root_dir, "疑似(Suspected)")
        copy_files(rejected_df, args.copy_rejected_to, args.source_root_dir, "拒绝(Rejected)")
    elif args.copy_clean_to or args.copy_suspected_to or args.copy_rejected_to:
        logger.warning("警告: 检测到 --copy-to-... 目录，但 --source-root-dir 未提供，跳过所有复制操作。")

def main():
    parser = argparse.ArgumentParser(description="步骤6: 根据VAD结果对噪声进行三分类、统计并归档")
    # --- 输入 ---
    parser.add_argument("--input-scp", required=True, help="输入的完整SCP (来自步骤5)")
    # --- VAD 三分类参数 ---
    parser.add_argument("--vad-models", nargs='+', required=True, help="用于三分类计数的VAD模型列表 (必须与步骤4一致)")
    parser.add_argument("--speech-ratio-threshold", type=float, required=True, help="人声比例阈值 (必须与步骤4一致)")
    # --- SCP 输出 ---
    parser.add_argument("--output-clean-scp", required=True, help="输出: 纯净(Clean)噪声的SCP")
    parser.add_argument("--output-suspected-scp", required=True, help="输出: 疑似(Suspected)噪声的SCP")
    parser.add_argument("--output-rejected-scp", required=True, help="输出: 拒绝(Rejected)噪声的SCP")
    # --- 报告输出 ---
    parser.add_argument("--stats-json-path", required=True, help="输出: 完整的统计 JSON 报告")
    parser.add_argument("--stats-txt-path", required=True, help="输出: 易读的统计 TXT 报告")
    # --- 复制 (可选) ---
    parser.add_argument("--source-root-dir", help="复制功能所需：所有待复制音频的共同根目录 (通常是步骤2或步骤3的输出目录)")
    parser.add_argument("--copy-clean-to", help="(可选) 将纯净噪声按类别复制到此目录")
    parser.add_argument("--copy-suspected-to", help="(可选) 将疑似含人声的噪声按类别复制到此目录")
    parser.add_argument("--copy-rejected-to", help="(可选) 将确认含人声的噪声按类别复制到此目录")
    # --- 日志 ---
    parser.add_argument("--log-dir", required=True, help="日志保存目录")
    
    args = parser.parse_args()
    logger = setup_logger(args.log_dir)
    
    # 检查复制参数是否完整
    if (args.copy_clean_to or args.copy_suspected_to or args.copy_rejected_to) and not args.source_root_dir:
        logger.warning("警告: 检测到 --copy-...-to 参数，但 --source-root-dir 未提供。将只进行统计，不执行复制。")
        
    try:
        run_screener_stats_and_copy(args, logger)
    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}", exc_info=True); raise
        
if __name__ == "__main__":
    main()