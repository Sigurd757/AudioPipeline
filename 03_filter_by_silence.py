#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 03: 按静音比例过滤并缩短长静音

本脚本是音频处理流程的第三步，承接 `02_split_audio_chunks.py` 的输出。

功能特性:
1.  读取步骤 02 生成的 `wav.scp` (含表头)。
2.  使用 `librosa.effects.split` 对每个音频块进行静音分析。
3.  过滤 (REJECTED):
    a.  音频包含 NaN (无效值)。
    b.  音频完全为 0 (全静音)。
    c.  音频的总静音比例超过设定的阈值 (`--ratio-threshold`)。
4.  编辑 (KEPT & modified):
    a.  如果音频块的总静音比例在阈值内，但包含超过 `--long-silence-threshold` 的长静音段。
    b.  脚本会将这些长静音段缩短为介于 `--shortened-min` 和 `--shortened-max` 之间的随机长度。
    c.  修改后的音频会保存到新的 `--output-dir` 目录中，保持相对路径结构。
5.  保留 (KEPT & not modified):
    a.  音频块通过了所有检查，且不包含需要缩短的长静音段。
6.  输出:
    a.  `--full-scp`: 包含所有输入音频块的处理结果 (KEPT/REJECTED, 原因, 静音比例等) 的完整报告。
    b.  `--filtered-scp`: 仅包含状态为 "KEPT" 的音频块，用于流程的下一步。
"""

import os
import csv
import soundfile as sf
import numpy as np
import librosa  # 用于静音检测
import argparse
import logging
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 用于向多进程工作函数传递全局配置
PROCESS_CONFIG = {}

def setup_logger(log_dir, log_name="03_silence_processing.log"):
    """
    配置并返回一个日志记录器 (Logger)。
    (此函数与 01, 02 脚本中的功能相同)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger("silence_processor")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# ==============================================================================
# 核心修改点 1: 工作函数现在处理来自输入SCP的 *完整行*
# ==============================================================================
def process_single_file(input_row):
    """
    工作函数：处理单个音频块的静音分析、过滤和编辑。
    
    此函数会接收来自步骤02的SCP文件中的一整行 (list) 作为参数，
    并返回一个追加了分析结果的新列表。

    参数:
        input_row (list): 来自输入SCP的一行数据 (例如 ['chunk_id_001', 'audio_id_01', 'path/to/chunk.wav', '10.0'])

    返回:
        list: input_row 加上新的分析列 [..., status, reason, silence_ratio, final_path]
    """
    # 从输入的行中解析出需要的信息
    # 根据 02_split_audio_chunks.py 的输出, chunk_id 在索引0, chunk_path 在索引2
    chunk_id = input_row[0]
    original_path = input_row[2]
    
    try:
        cfg = PROCESS_CONFIG
        y, sr = sf.read(original_path, dtype='float32')
        total_duration = len(y) / sr

        # -------------------------------------
        # 1. 完整性检查 (过滤)
        # -------------------------------------
        if np.isnan(y).any():
            # 如果音频包含无效的 NaN 值
            return input_row + ["REJECTED", "contains_nan", -1.0, original_path]
        if np.all(y == 0):
            # 如果音频所有采样点都为 0
            return input_row + ["REJECTED", "all_zeros", 1.0, original_path]

        # -------------------------------------
        # 2. 总体静音比例分析 (过滤)
        # -------------------------------------
        # librosa.effects.split 返回非静音部分的 (start, end) 采样点索引
        # top_db 是相对于音频最大分贝的阈值，用户传入的 cfg['silence_db_threshold'] 是正数，这里取反
        non_silent_intervals = librosa.effects.split(y, top_db=abs(cfg['silence_db_threshold']))
        # 计算所有非静音部分的总时长
        non_silent_duration = sum((end - start) / sr for start, end in non_silent_intervals)
        total_silence_duration = total_duration - non_silent_duration
        silence_ratio = total_silence_duration / total_duration if total_duration > 0 else 0

        # 如果总静音比例超过了设定的阈值 (例如 90%)
        if silence_ratio > cfg['total_silence_ratio_threshold']:
            return input_row + ["REJECTED", f"silence_ratio_{silence_ratio:.2f}", f"{silence_ratio:.4f}", original_path]

        # -------------------------------------
        # 3. 长静音段缩减 (编辑)
        # -------------------------------------
        modified = False  # 标记此文件是否被修改过
        final_audio_segments = [] # 存储重构音频的片段
        
        # 这个逻辑通过遍历非静音区间 (non_silent_intervals) 来重构音频，
        # 并在此过程中检查和替换非静音区间 *之间* 的长静音。
        
        current_pos = 0 # 记录当前处理到的采样点位置
        for start, end in non_silent_intervals:
            # 检查 current_pos 和 non_silent 区间开始 (start) 之间的静音
            silence_start = current_pos
            silence_end = start
            if silence_end > silence_start: # 如果存在静音
                silence_duration = (silence_end - silence_start) / sr
                if silence_duration > cfg['long_silence_duration_threshold']:
                    # 发现长静音, 进行缩短
                    modified = True
                    # 缩短为 [min, max] 之间的一个随机时长
                    new_len_sec = random.uniform(cfg['shortened_silence_min'], cfg['shortened_silence_max'])
                    final_audio_segments.append(np.zeros(int(new_len_sec * sr), dtype=np.float32))
                else:
                    # 静音不长, 保持原样
                    final_audio_segments.append(y[silence_start:silence_end])
            
            # 添加非静音部分
            final_audio_segments.append(y[start:end])
            current_pos = end # 更新位置到当前非静音部分的末尾

        # 处理最后一个非静音区间 *之后* 的尾部静音
        if current_pos < len(y):
            silence_duration = (len(y) - current_pos) / sr
            if silence_duration > cfg['long_silence_duration_threshold']:
                # 发现长静音, 进行缩短
                modified = True
                new_len_sec = random.uniform(cfg['shortened_silence_min'], cfg['shortened_silence_max'])
                final_audio_segments.append(np.zeros(int(new_len_sec * sr), dtype=np.float32))
            else:
                # 静音不长, 保持原样
                final_audio_segments.append(y[current_pos:])
        
        # -------------------------------------
        # 4. 保存与返回 (KEPT)
        # -------------------------------------
        final_path = original_path # 默认最终路径为原始路径
        if modified:
            # 如果音频被修改了 (长静音被缩短)
            # 1. 合并所有音频片段
            y_modified = np.concatenate(final_audio_segments) if final_audio_segments else np.array([], dtype=np.float32)
            
            # 2. 计算新文件的保存路径，保持相对目录结构
            relative_path = os.path.relpath(original_path, cfg['chunks_dir_base'])
            final_path = os.path.join(cfg['output_dir'], relative_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            # 3. 写入修改后的文件
            sf.write(final_path, y_modified, sr, subtype='PCM_16')
        
        # 返回 "KEPT" 状态
        return input_row + ["KEPT", f"modified_{modified}", f"{silence_ratio:.4f}", final_path]

    except Exception as e:
        # 捕获读取或处理中的其他异常
        return input_row + ["REJECTED", f"error_{e}", -1.0, original_path]

def main():
    parser = argparse.ArgumentParser(description='步骤3: 过滤静音音频并缩减长静音段')
    parser.add_argument('--input-scp', required=True, help='输入的 wav.scp (来自步骤2)')
    parser.add_argument('--output-dir', required=True, help='存放修改后音频的根目录 (如果发生缩减)')
    parser.add_argument('--chunks-dir', required=True, help='原始音频块的根目录 (用于计算相对路径)')
    parser.add_argument('--full-scp', required=True, help='输出的完整分析结果 scp (tsv, 含表头)')
    parser.add_argument('--filtered-scp', required=True, help='输出的筛选后 scp (tsv, 含表头, 用于下一步)')
    parser.add_argument('--log-dir', required=True, help='日志保存目录')
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--silence-db', type=float, default=50.0, help='静音检测的分贝阈值 (正数, e.g., 50 表示 -50dBFS 以下为静音)')
    parser.add_argument('--ratio-threshold', type=float, default=0.9, help='总静音比例过滤阈值 (e.g., 0.9 表示静音超过90%则丢弃)')
    parser.add_argument('--long-silence-threshold', type=float, default=2.0, help='长静音段的时长阈值(秒), 超过则缩减')
    parser.add_argument('--shortened-min', type=float, default=0.5, help='缩减后静音段的最小长度(秒)')
    parser.add_argument('--shortened-max', type=float, default=1.5, help='缩减后静音段的最大长度(秒)')
    
    args = parser.parse_args()
    logger = setup_logger(args.log_dir)
    logger.info("开始步骤3: 静音处理...")
    logger.info(f"静音阈值: -{args.silence_db} dBFS")
    logger.info(f"静音比例过滤阈值: > {args.ratio_threshold * 100:.0f}%")
    logger.info(f"长静音缩减阈值: > {args.long_silence_threshold}s")

    # 1. 设置全局配置
    global PROCESS_CONFIG
    PROCESS_CONFIG = {
        'output_dir': args.output_dir,
        'chunks_dir_base': args.chunks_dir,
        'silence_db_threshold': args.silence_db,
        'total_silence_ratio_threshold': args.ratio_threshold,
        'long_silence_duration_threshold': args.long_silence_threshold,
        'shortened_silence_min': args.shortened_min,
        'shortened_silence_max': args.shortened_max
    }

    # ==============================================================================
    # 核心修改点 2: 读取完整的行和表头
    # ==============================================================================
    files_to_process = [] # 存储待处理的 *行*
    input_header = []     # 存储输入SCP的 *表头*
    try:
        with open(args.input_scp, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            input_header = next(reader) # 读取表头
            for row in reader:
                files_to_process.append(row) # 添加完整行
    except FileNotFoundError:
        logger.error(f"输入文件未找到: {args.input_scp}")
        return
    except Exception as e:
        logger.error(f"读取SCP文件失败: {e}")
        return

    logger.info(f"共 {len(files_to_process)} 个音频块待处理。使用 {args.workers} 个进程。")

    # 2. 启动多进程池
    all_results = [] # 存储所有 (包含新列的) 结果行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, item) for item in files_to_process]
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="静音处理进度"):
            all_results.append(future.result())

    # ==============================================================================
    # 核心修改点 3: 创建新表头并写入完整结果
    # ==============================================================================
    
    # 新表头 = 原始表头 + 新增的分析列名
    output_header = input_header + ['status', 'reason', 'silence_ratio', 'final_path']

    # 3. 写入 full.scp (完整报告)
    os.makedirs(os.path.dirname(args.full_scp), exist_ok=True)
    with open(args.full_scp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(output_header)
        writer.writerows(all_results)
        
    # 4. 写入 filtered.scp (筛选后的清单)
    
    # 'status' 列的索引 = 原始表头的长度
    status_column_index = len(input_header)
    # 筛选出所有 status == "KEPT" 的行
    filtered_results = [res for res in all_results if res[status_column_index] == "KEPT"]
    
    os.makedirs(os.path.dirname(args.filtered_scp), exist_ok=True)
    with open(args.filtered_scp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(output_header) # 写入相同的表头
        writer.writerows(filtered_results)
        
    # 5. 结束日志
    logger.info("步骤3处理完成！")
    logger.info(f"总输入: {len(all_results)}, 保留 (KEPT): {len(filtered_results)}, 拒绝 (REJECTED): {len(all_results) - len(filtered_results)}")
    logger.info(f"完整分析报告: {args.full_scp}")
    logger.info(f"筛选后清单 (下一步输入): {args.filtered_scp}")

if __name__ == "__main__":
    if os.name != 'posix':
        multiprocessing.set_start_method('spawn', force=True)
    main()