#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 01: 音频重采样 (使用 FFmpeg)

本脚本用于遍历 JSON 配置文件中指定的多个数据集目录，
查找所有支持的音频格式文件，并使用 FFmpeg 将它们批量转换为
具有统一采样率、通道数和格式 (16-bit PCM WAV) 的新文件。

功能特性:
1.  读取 JSON 配置文件以获取数据集名称和路径。
2.  递归扫描所有数据集路径，找出支持的音频文件 (如 .wav, .flac, .mp3)。
3.  使用多进程 (`concurrent.futures.ProcessPoolExecutor`) 并行处理文件。
4.  依赖外部工具 FFmpeg 执行实际的重采样和格式转换，具有高鲁棒性。
5.  在目标目录中按 `dataset_name/relative_path/` 结构保存新文件。
6.  生成一个 `wav.scp` (制表符分隔, 含 header)，记录 audio_id, 原始路径, 和处理后路径。
"""

import os
import json
import argparse
import logging
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess # 导入用于执行外部命令的库

# 全局变量，用于向多进程工作函数 (process_single_file_ffmpeg) 传递配置
PROCESS_CONFIG = {}

def setup_logger(log_dir, log_name="01_resample.log"):
    """
    配置并返回一个日志记录器 (Logger)。

    将日志同时输出到控制台和指定的日志文件。
    (此函数与 02_split_audio_chunks.py 中的功能相同)
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger("audio_resampler")
    logger.setLevel(logging.INFO)
    # 如果 logger 已经有 handlers，说明已配置过，直接返回
    if logger.handlers: return logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def generate_audio_id(dataset_name, root, file, dataset_path):
    """
    根据数据集名称和文件相对路径生成一个唯一的 audio_id。
    
    格式: {dataset_name}_{relative_path_with_underscores}_{file_name_without_ext}
    例如: my_dataset_train_clean_speaker1_file001
    """
    relative_path = os.path.relpath(root, dataset_path)
    file_name = os.path.splitext(file)[0]
    
    # 替换路径分隔符为下划线，并过滤掉空字符串 (例如当 relative_path 为 '.')
    id_parts = [dataset_name, relative_path.replace(os.sep, '_'), file_name]
    return "_".join(filter(None, id_parts))

# ==============================================================================
# 核心修改：使用 FFmpeg 替代 Librosa
# ==============================================================================
def process_single_file_ffmpeg(args):
    """
    工作函数：使用 FFmpeg 命令处理单个音频文件。

    执行重采样、通道转换、格式转换为 16-bit PCM WAV。
    从全局 PROCESS_CONFIG 获取配置。

    参数:
        args (tuple): 包含 (dataset_name, dataset_path, root, file) 的元组。

    返回:
        tuple: (success, audio_id, raw_path, processed_path, error_msg)
            - success (bool): 处理是否成功。
            - audio_id (str | None): 成功时返回生成的 audio_id。
            - raw_path (str): 原始输入文件路径。
            - processed_path (str | None): 成功时返回新文件的保存路径。
            - error_msg (str | None): 失败时返回错误信息。
    """
    dataset_name, dataset_path, root, file = args
    input_file_path = os.path.join(root, file)
    
    try:
        # 1. 解析全局配置
        cfg = PROCESS_CONFIG
        
        # 2. 检查文件格式是否支持
        ext = os.path.splitext(file)[-1].lower().strip('.')
        if ext not in cfg['supported_formats']:
            return (False, None, None, None, f"不支持的格式: {ext}")
            
        # 3. 构建输出路径和 audio_id
        relative_path = os.path.relpath(root, dataset_path)
        file_name_wav = os.path.splitext(file)[0] + '.wav'
        # 目标路径格式: <target_root>/<dataset_name>/<relative_path>/<file_name>.wav
        output_file_path = os.path.join(cfg['target_root'], dataset_name, relative_path, file_name_wav)
        audio_id = generate_audio_id(dataset_name, root, file, dataset_path)
        
        # 4. 确保输出目录存在 (原子操作，多进程安全)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # 5. 构建 FFmpeg 命令
        command = [
            'ffmpeg',
            '-i', input_file_path,      # 输入文件
            '-y',                       # 覆盖已存在的文件 (non-interactive)
            '-ar', str(cfg['target_sr']), # 设置目标采样率
            '-ac', str(cfg['target_channels']), # 设置目标通道数 (e.g., 1 for mono)
            '-c:a', 'pcm_s16le',        # 设置输出编码为 16-bit PCM (little-endian)
            '-hide_banner',             # 隐藏不必要的启动信息
            '-loglevel', 'error',       # 只在发生错误时打印日志
            output_file_path            # 输出文件
        ]
        
        # 6. 执行命令
        # capture_output=True 捕获 stdout 和 stderr
        # text=True 将 stdout/stderr 解码为文本
        # check=False 防止在 returncode != 0 时抛出异常，我们稍后手动检查
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # 7. 检查 FFmpeg 执行结果
        if result.returncode != 0:
            # 如果失败，记录 ffmpeg 的标准错误输出
            error_message = result.stderr.strip()
            return (False, None, input_file_path, None, f"FFmpeg error: {error_message}")

        # 8. 成功返回
        return (True, audio_id, input_file_path, output_file_path, None)

    except Exception as e:
        # 捕获 Python 层的异常 (例如路径问题)
        return (False, None, input_file_path, None, str(e))

def collect_all_files(input_datasets, supported_formats):
    """
    遍历所有数据集路径，收集所有支持格式的音频文件。

    参数:
        input_datasets (dict): {dataset_name: path} 的字典。
        supported_formats (set): 支持的文件扩展名集合 (如 {'wav', 'flac'})。

    返回:
        list: 待处理文件参数的列表
              [(dataset_name, dataset_path, root, file), ...]
    """
    files_to_process = []
    for dataset_name, dataset_path in input_datasets.items():
        if not os.path.exists(dataset_path):
            # 使用 logging.warning，因为这是在主进程中调用的
            logging.warning(f"数据集路径不存在: {dataset_path}，将跳过该数据集")
            continue
        # 遍历数据集目录
        for root, _, files in os.walk(dataset_path):
            for file in files:
                ext = file.split('.')[-1].lower()
                if ext in supported_formats:
                    # 添加任务参数元组
                    files_to_process.append((dataset_name, dataset_path, root, file))
    return files_to_process

def load_dataset_config(config_path):
    """
    加载并校验 JSON 格式的数据集配置文件。
    
    配置文件应为 {"dataset_name_1": "path/to/dataset_1", ...} 格式。
    
    参数:
        config_path (str): JSON 配置文件路径。
        
    返回:
        dict: 数据集配置字典。
        
    抛出:
        Exception: 如果文件读取或格式校验失败。
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
        if not isinstance(config, dict): raise ValueError("数据集配置文件必须是JSON对象（字典格式）")
        for name, path in config.items():
            if not isinstance(path, str): raise ValueError(f"数据集 {name} 的路径必须是字符串")
        return config
    except Exception as e: 
        raise Exception(f"加载数据集配置失败: {str(e)}")


def process_datasets(input_config, target_root, scp_path, log_dir, 
                     supported_formats={'wav', 'flac', 'pcm', 'mp3'},
                     target_sr=16000, target_channels=1, target_subtype='PCM_16',
                     max_workers=None):
    """
    并行处理所有数据集的主函数。

    (此函数已修改为使用 process_single_file_ffmpeg)
    
    参数:
        input_config (dict): 从 JSON 加载的数据集配置。
        target_root (str): 保存新文件的根目录。
        scp_path (str): 输出 wav.scp 文件的路径。
        log_dir (str): 日志目录。
        supported_formats (set): 支持的音频格式。
        target_sr (int): 目标采样率。
        target_channels (int): 目标通道数 (1=mono, 2=stereo)。
        target_subtype (str): (此参数在此FFmpeg版本中未使用，但保留)
        max_workers (int | None): 最大并行进程数，None表示使用CPU核心数。
    """
    logger = setup_logger(log_dir)
    logger.info("开始使用 FFmpeg 并行音频重采样...")
    # 打印配置信息 (日志)
    logger.info(f"目标根目录: {target_root}")
    logger.info(f"输出 SCP: {scp_path}")
    logger.info(f"目标采样率: {target_sr}Hz, 通道数: {target_channels}")
    logger.info(f"支持的格式: {supported_formats}")
    
    # 1. 设置全局配置，供子进程访问
    global PROCESS_CONFIG
    PROCESS_CONFIG = {
        'target_sr': target_sr,
        'target_channels': target_channels,
        'target_subtype': target_subtype, # (此参数在FFmpeg中硬编码为 pcm_s16le)
        'target_root': target_root,
        'supported_formats': supported_formats
    }
    
    # 2. 收集所有文件
    files_to_process = collect_all_files(input_config, supported_formats)
    total_files = len(files_to_process)
    logger.info(f"共检测到 {total_files} 个音频文件待处理")
    
    if total_files == 0:
        logger.info("没有需要处理的文件，退出程序")
        return
        
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    logger.info(f"使用 {max_workers} 个进程并行处理")
    
    # 确保 scp 文件的父目录存在
    os.makedirs(os.path.dirname(scp_path), exist_ok=True)
    
    success_files, failed_files, results = 0, 0, []
    
    # 3. 启动多进程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务，使用新的 ffmpeg 处理函数
        futures = [executor.submit(process_single_file_ffmpeg, args) for args in files_to_process]
        
        progress_bar = tqdm(total=total_files, desc="FFmpeg处理进度")
        
        # 4. 收集处理结果
        for future in as_completed(futures):
            progress_bar.update(1)
            result = future.result()
            results.append(result) # 无论成功失败，都先收集
            
            success, audio_id, raw_path, processed_path, error = result
            if success:
                success_files += 1
            else:
                failed_files += 1
                # 在子进程中发生的错误，通过日志记录
                logger.error(f"处理失败: {raw_path}，错误: {error}")
    
    progress_bar.close()
    
    # 5. 写入 wav.scp 文件 (只写入成功的条目)
    with open(scp_path, 'w', newline='', encoding='utf-8') as scp_file:
        scp_writer = csv.writer(scp_file, delimiter='\t')
        # 写入表头
        scp_writer.writerow(['audio_id', 'raw_path', 'processed_path'])
        for result in results:
            success, audio_id, raw_path, processed_path, _ = result
            if success:
                # 写入: audio_id, 原始路径, 处理后路径
                scp_writer.writerow([audio_id, raw_path, processed_path])
    
    # 6. 打印最终总结日志
    logger.info("处理完成！")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功处理并写入scp: {success_files}")
    logger.info(f"处理失败: {failed_files}")
    logger.info(f"成功率: {success_files/total_files*100:.2f}%" if total_files > 0 else "无文件处理")


def main():
    """
    程序入口：解析命令行参数并启动数据集处理。
    """
    parser = argparse.ArgumentParser(description='支持JSON配置的并行音频重采样脚本(FFmpeg版)')
    parser.add_argument('--target', required=True, help='目标根目录路径 (存放新生成wav文件的位置)')
    parser.add_argument('--scp', required=True, help='生成的wav.scp文件路径 (tsv格式)')
    parser.add_argument('--config', required=True, help='数据集配置JSON文件路径 ({"dataset_name": "path"})')
    parser.add_argument('--log-dir', default='./resampling_logs', help='日志保存目录')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数，默认使用CPU核心数')
    parser.add_argument('--sr', type=int, default=16000, help='目标采样率，默认16000')
    parser.add_argument('--formats', nargs='+', default=['wav', 'flac', 'pcm', 'mp3'], help='支持的音频格式 (空格分隔)')
    args = parser.parse_args()
    
    try:
        # 加载数据集配置
        input_datasets = load_dataset_config(args.config)
    except Exception as e:
        print(f"配置文件错误: {str(e)}")
        return
    
    # 启动主处理流程
    process_datasets(
        input_config=input_datasets,
        target_root=args.target,
        scp_path=args.scp,
        log_dir=args.log_dir,
        supported_formats=set(args.formats), # 转换为 set 以提高查找效率
        target_sr=args.sr,
        max_workers=args.workers
    )

if __name__ == "__main__":
    # 兼容 Windows 和 macOS (使用 'spawn')：
    # 在非 Posix 系统 (如 Windows) 上使用 'spawn' 启动方法是必要的，
    # 以避免多进程在子进程中重新导入主脚本时出现问题。
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn')
    main()