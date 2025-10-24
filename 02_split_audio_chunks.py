#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 02: 音频切分

本脚本用于将 `wav.scp` 文件中列出的长音频文件切分为固定时长的音频块。

功能特性:
1.  读取一个 `wav.scp` (制表符分隔, 包含 header) 作为输入。
2.  将每个音频文件按指定的 `chunk-len` (秒) 切分。
3.  处理最后一段音频：如果其时长大于等于 `min-last-chunk` (秒)，则保留；否则丢弃。
4.  检查并确保所有音频的采样率与 `--sr` 参数一致。
5.  将非单声道音频转换为单声道（取平均）。
6.  使用多进程 (`concurrent.futures.ProcessPoolExecutor`) 加速处理。
7.  输出一个新的 `wav.scp` 文件，包含所有成功生成的音频块信息 (chunk_id, original_audio_id, chunk_path, duration)。
8.  将切分后的 `*.wav` 文件保存到指定的输出目录，并根据 `audio_id` 的第一部分 (用 '_' 分隔) 创建子目录。
"""

import os
import csv
import soundfile as sf
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing

# 用于向多进程工作函数 (process_single_file) 传递全局配置的字典
PROCESS_CONFIG = {}

def setup_logger(log_dir, log_name="02_splitting.log"):
    """
    配置并返回一个日志记录器 (Logger)。

    将日志同时输出到控制台和指定的日志文件。

    参数:
        log_dir (str): 日志文件存放的目录。
        log_name (str): 日志文件的名称。

    返回:
        logging.Logger: 配置好的日志记录器实例。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    
    logger = logging.getLogger("audio_splitter")
    logger.setLevel(logging.INFO)
    
    # 如果 logger 已经有 handlers，说明已配置过，直接返回
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def process_single_file(args):
    """
    工作函数：处理单个音频文件的切分任务。

    从全局 PROCESS_CONFIG 获取配置。

    参数:
        args (tuple): 包含 (audio_id, input_path) 的元组。

    返回:
        tuple: (success, message, results)
            - success (bool): 处理是否成功。
            - message (str | None): 如果失败，返回错误信息。
            - results (list): 包含多个音频块信息列表的列表，格式为
                              [[chunk_id, audio_id, output_path, duration], ...]
    """
    audio_id, input_path = args
    results = []  # 存储此文件成功切分出的所有块信息
    
    try:
        # 从全局配置中获取参数
        chunk_len_sec = PROCESS_CONFIG['chunk_len_sec']
        output_dir = PROCESS_CONFIG['output_dir']
        sample_rate = PROCESS_CONFIG['sample_rate']
        min_last_chunk_len = PROCESS_CONFIG['min_last_chunk_len']
        
        # 1. 读取音频文件
        y, sr = sf.read(input_path, dtype='float32')
        
        # 2. 检查采样率
        if sr != sample_rate:
            return (False, f"采样率不匹配: 文件 {input_path} 为 {sr}Hz, 期望 {sample_rate}Hz", [])

        # 3. 转换为单声道 (如果需要)
        if y.ndim > 1:
            y = y.mean(axis=1)

        total_samples = len(y)
        chunk_samples = int(chunk_len_sec * sr) # 每个完整块的采样点数
        
        # --- 切分逻辑开始 ---
        
        # 4. 计算并处理所有完整的块
        num_full_chunks = math.floor(total_samples / chunk_samples)
        
        # 根据 audio_id (例如 "F0001_001") 的第一部分 ("F0001") 创建子目录
        sub_dir_name = audio_id.split('_')[0]
        
        for i in range(num_full_chunks):
            start_sample = i * chunk_samples
            end_sample = start_sample + chunk_samples
            chunk_data = y[start_sample:end_sample]
            
            chunk_id = f"{audio_id}_chunk{i:04d}"
            output_chunk_dir = os.path.join(output_dir, sub_dir_name)
            os.makedirs(output_chunk_dir, exist_ok=True)
            output_path = os.path.join(output_chunk_dir, f"{chunk_id}.wav")
            
            # 写入 16-bit PCM 格式的 wav 文件
            sf.write(output_path, chunk_data, sr, subtype='PCM_16')
            
            duration = len(chunk_data) / sr
            results.append([chunk_id, audio_id, output_path, f"{duration:.3f}"])

        # 5. 处理最后一段不足时长的音频
        start_of_remainder = num_full_chunks * chunk_samples
        remainder_samples = total_samples - start_of_remainder
        
        # 如果存在剩余部分
        if remainder_samples > 0:
            remainder_duration_sec = remainder_samples / sr
            
            # 检查剩余部分的时长是否大于等于最小阈值
            if remainder_duration_sec >= min_last_chunk_len:
                # 保留这最后一段
                last_chunk_data = y[start_of_remainder:]
                
                chunk_id = f"{audio_id}_chunk{num_full_chunks:04d}" # 序号顺延
                output_chunk_dir = os.path.join(output_dir, sub_dir_name)
                os.makedirs(output_chunk_dir, exist_ok=True)
                output_path = os.path.join(output_chunk_dir, f"{chunk_id}.wav")

                sf.write(output_path, last_chunk_data, sr, subtype='PCM_16')
                
                duration = len(last_chunk_data) / sr
                results.append([chunk_id, audio_id, output_path, f"{duration:.3f}"])
            # else: (隐式) 如果小于阈值，则丢弃这最后一段

        # --- 切分逻辑结束 ---

        return (True, None, results)

    except Exception as e:
        # 捕获所有可能的异常，确保进程不会崩溃
        return (False, f"处理 {input_path} 时发生严重错误: {str(e)}", [])

def main():
    """
    主函数：解析命令行参数，读取输入 scp，分发多进程任务，并收集结果写入输出 scp。
    """
    parser = argparse.ArgumentParser(description='将音频切分为固定长度的块，并根据阈值保留最后一段')
    parser.add_argument('--input-scp', required=True, help='输入的 wav.scp 文件路径 (TSV格式, 含header)')
    parser.add_argument('--output-dir', required=True, help='存放切分后音频块的根目录')
    parser.add_argument('--output-scp', required=True, help='输出新的 wav.scp 文件路径')
    parser.add_argument('--chunk-len', type=float, default=10.0, help='每个完整音频块的目标长度（秒）')
    # 新增的命令行参数
    parser.add_argument('--min-last-chunk', type=float, default=2.0, help='最后一段音频如果大于等于此时长（秒），则保留')
    parser.add_argument('--log-dir', required=True, help='日志保存目录')
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--sr', type=int, required=True, help='期望的音频采样率 (将用于校验)')
    
    args = parser.parse_args()
    
    # 1. 初始化日志
    logger = setup_logger(args.log_dir)
    logger.info(f"开始音频切分处理...")
    logger.info(f"切分时长: {args.chunk_len}s, 最后一段最小时长: {args.min_last_chunk}s")
    logger.info(f"目标采样率: {args.sr}Hz")
    
    # 2. 设置全局配置，用于传递给子进程
    global PROCESS_CONFIG
    PROCESS_CONFIG = {
        'chunk_len_sec': args.chunk_len,
        'output_dir': args.output_dir,
        'sample_rate': args.sr,
        'min_last_chunk_len': args.min_last_chunk
    }
    
    # 3. 读取输入的 scp 文件
    files_to_process = [] # 待处理的任务列表 [(audio_id, input_path), ...]
    try:
        with open(args.input_scp, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader) # 跳过表头
            # 假设 scp 格式为: audio_id, ... , path
            # 这里我们取第0列为 audio_id, 第2列为 path (根据原始代码 row[0], row[2])
            for row in reader:
                files_to_process.append((row[0], row[2])) 
    except FileNotFoundError:
        logger.error(f"输入文件未找到: {args.input_scp}")
        return
    except Exception as e:
        logger.error(f"读取输入scp文件失败: {e}")
        return

    total_files = len(files_to_process)
    logger.info(f"共发现 {total_files} 个文件待切分。使用 {args.workers} 个进程。")

    all_chunk_results = [] # 汇总所有进程返回的成功结果
    failed_count = 0       # 统计处理失败的文件数
    
    # 4. 启动多进程池
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_file, item) for item in files_to_process]
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=total_files, desc="切分进度"):
            success, message, results = future.result()
            if success:
                if results: # results 可能为空列表 (例如音频总时长小于 min_last_chunk)
                    all_chunk_results.extend(results)
            else:
                failed_count += 1
                logger.error(message) # 记录失败原因
    
    # 5. 将所有成功的结果写入新的 scp 文件
    os.makedirs(os.path.dirname(args.output_scp), exist_ok=True)
    with open(args.output_scp, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # 写入表头
        writer.writerow(['chunk_id', 'original_audio_id', 'chunk_path', 'duration'])
        if all_chunk_results:
            writer.writerows(all_chunk_results)
        
    # 6. 结束日志
    logger.info("切分处理完成！")
    logger.info(f"原始文件数: {total_files}, 处理失败: {failed_count}")
    logger.info(f"成功生成的音频块总数: {len(all_chunk_results)}")
    logger.info(f"新清单文件已保存至: {args.output_scp}")

if __name__ == "__main__":
    # 确保在非 Posix 系统 (如 Windows) 上使用 multiprocessing 时
    # 采用 'spawn' 启动方式，避免子进程相关问题
    if os.name != 'posix':
        multiprocessing.set_start_method('spawn', force=True)
    main()