#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 04: VAD 语音活动检测与过滤

本脚本是音频处理流程的第四步，承接 `03_filter_by_silence.py` 的输出。

核心目标：
遍历所有在上一步中 "KEPT" 的音频块，使用一个或多个 VAD 模型进行人声检测。
如果 *任何* 一个 VAD 模型检测到的人声比例超过阈值，则该音频块被视为
"REJECTED" (包含人声)。
目的是筛选出纯净的噪声片段。

功能特性:
1.  支持多种VAD模型，并通过命令行参数 `--vad-models` 灵活选择使用：
    - 'silero': Silero VAD (https://github.com/snakers4/silero-vad)
    - 'funasr': FunASR FSMN-VAD (https://github.com/alibaba-damo-academy/FunASR)
    - 'webrtc': Google WebRTC VAD (需 `webrtcvad` 库)
    - 'nemo_onnx': NeMo Marblenet VAD (需提供预导出的 ONNX 模型)
2.  使用抽象基类 `BaseVAD` 统一不同 VAD 模型的接口。
3.  自动预下载和缓存 Silero 和 FunASR 模型。
4.  使用多进程并行处理，并在每个子进程中独立初始化 VAD 模型实例。
5.  使用 Pandas 库高效读写 TSV(SCP) 文件，确保在处理包含特殊字符的
    路径时（如JSON字符串）不会出错。
6.  输出:
    a.  `--full-scp`: 包含所有输入音频块的处理结果 (KEPT/REJECTED) 和
        每个VAD模型的详细检测比例 (JSON格式)。
    b.  `--filtered-scp`: 仅包含状态为 "KEPT" (即纯净噪声) 的音频块。
"""

import os
import csv
import soundfile as sf
import numpy as np
import torch
import torchaudio
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
from silero_vad import load_silero_vad, get_speech_timestamps # Silero VAD 依赖
import librosa # 用于音频重采样
import onnxruntime # NeMo ONNX VAD 依赖
import pandas as pd # 用于读写 TSV 文件

# --- NeMo预处理器 (独立实现) ---
# 这个类是 NeMo VAD ONNX 模型所必需的特征提取器
class NeMoMelPreprocessor(torch.nn.Module):
    """
    一个独立的PyTorch模块，用于复现 NeMo VAD 所需的 Mel 谱图预处理。
    这确保了在没有安装 NeMo 工具包的情况下也能正确使用 ONNX 模型。
    """
    def __init__(self, sr=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, window='hann'):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel 谱图转换器
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels, sample_rate=self.sr, n_stft=self.n_fft // 2 + 1
        )
        # 注册窗函数为 buffer
        self.register_buffer('window', torch.hann_window(self.win_length, periodic=False))

    def forward(self, audio_signal):
        """
        前向传播：将原始音频波形 (audio_signal) 转换为 Log-Mel 谱图。
        """
        # 1. 计算 STFT
        stft = torch.stft(
            audio_signal, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, center=True, window=self.window.to(audio_signal.device),
            return_complex=True,
        )
        # 2. 计算幅度平方 (功率谱)
        magnitudes = stft.abs().pow(2)
        # 3. 转换为 Mel 谱
        mel_spec = self.mel_scale(magnitudes)
        # 4. 取 Log (加上一个很小的数防止 log(0))
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-9))
        return log_mel_spec

# --- VAD 抽象类和实现 ---

class BaseVAD:
    """VAD 模型的抽象基类，统一定义接口。"""
    def __init__(self, *args, **kwargs):
        """构造函数：调用 load_model 来初始化模型实例。"""
        self.model = self.load_model(*args, **kwargs)
    
    def load_model(self, *args, **kwargs):
        """[抽象方法] 加载 VAD 模型。子类必须实现此方法。"""
        raise NotImplementedError
        
    def get_speech_ratio(self, y: np.ndarray, sr: int) -> float:
        """
        [抽象方法] 计算给定音频 (y, sr) 中的语音活动比例。
        
        返回:
            float: 语音活动比例 (0.0 到 1.0 之间)。
        """
        raise NotImplementedError

class SileroVAD(BaseVAD):
    """Silero VAD 实现"""
    def load_model(self, *args, **kwargs):
        """加载 Silero VAD 模型 (会使用 PyTorch Hub 自动下载)。"""
        return load_silero_vad()
        
    def get_speech_ratio(self, y: np.ndarray, sr: int) -> float:
        """计算 Silero VAD 的语音比例。"""
        # Silero VAD 必须使用 16kHz
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        audio_tensor = torch.from_numpy(y)
        # 获取语音时间戳 (单位: 采样点)
        speech_ts = get_speech_timestamps(audio_tensor, self.model, sampling_rate=sr)
        # 计算总的语音采样点数
        speech_samples = sum(d['end'] - d['start'] for d in speech_ts)
        # 返回语音时长占总时长的比例
        return (speech_samples / sr) / (len(y) / sr) if len(y) > 0 else 0

class FunASRVAD(BaseVAD):
    """FunASR VAD 实现"""
    def load_model(self, *args, **kwargs):
        """加载 FunASR VAD 模型 (会自动下载)。"""
        from funasr import AutoModel
        return AutoModel(model="fsmn-vad", model_revision="v2.0.4", device="cpu", disable_update=True, log_level="ERROR")
        
    def get_speech_ratio(self, y: np.ndarray, sr: int) -> float:
        """计算 FunASR VAD 的语音比例。"""
        # FunASR VAD 必须使用 16kHz
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        # FunASR 返回的时间戳单位是毫秒 (ms)
        result = self.model.generate(y, sample_rate=sr)
        # 将毫秒转换为秒
        speech_duration = sum(d['end'] - d['start'] for d in result[0].get('vad_timestamps', [])) / 1000.0
        return speech_duration / (len(y) / sr) if len(y) > 0 else 0

class WebRTCVAD(BaseVAD):
    """WebRTC VAD 实现 (基于 `webrtcvad` 库)"""
    def load_model(self, *args, **kwargs):
        """初始化 WebRTC VAD，可设置 'aggressiveness' (激进程度)。"""
        import webrtcvad
        return webrtcvad.Vad(kwargs.get('aggressiveness', 1))
        
    def get_speech_ratio(self, y: np.ndarray, sr: int) -> float:
        """计算 WebRTC VAD 的语音比例。"""
        # 1. 检查支持的采样率
        SUPPORTED_RATES = [8000, 16000, 32000, 48000]
        if sr not in SUPPORTED_RATES: y = librosa.resample(y, orig_sr=sr, target_sr=16000); sr=16000
        # 2. WebRTC VAD 处理 30ms 的帧
        frame_samples = int(sr * 30 / 1000)
        # 3. 必须使用 16-bit 整数
        y_int16 = np.int16(y * 32767); speech_frames, total_frames = 0, 0
        # 4. 逐帧处理
        for i in range(0, len(y_int16), frame_samples):
            frame = y_int16[i:i+frame_samples]
            if len(frame) < frame_samples: break
            total_frames += 1
            # 5. 检查帧是否为语音
            if self.model.is_speech(frame.tobytes(), sr): speech_frames += 1
        # 6. 返回语音帧的比例
        return speech_frames / total_frames if total_frames > 0 else 0

class NeMoONNXVAD(BaseVAD):
    """NeMo VAD (ONNX版) 实现"""
    def load_model(self, *args, **kwargs):
        """加载 ONNX session 和 NeMo 预处理器。"""
        onnx_path = kwargs.get('onnx_path')
        if not onnx_path: raise ValueError("NeMoONNXVAD 需要 onnx_path 参数")
        
        session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        preprocessor = NeMoMelPreprocessor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preprocessor = preprocessor.to(device)
        return {'session': session, 'preprocessor': preprocessor, 'device': device}

    def get_speech_ratio(self, y: np.ndarray, sr: int) -> float:
        """计算 NeMo VAD 的语音比例。"""
        session, preprocessor, device = self.model['session'], self.model['preprocessor'], self.model['device']
        # 1. 重采样到 16kHz
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # 2. 预处理：波形 -> Log-Mel 谱图
        input_signal = torch.from_numpy(y).unsqueeze(0).to(device)
        processed_signal = preprocessor(input_signal) # [B, M, T]
        
        # 3. 准备 ONNX 输入
        input_names = [inp.name for inp in session.get_inputs()]
        ort_inputs = {input_names[0]: processed_signal.cpu().numpy()}
        # 某些导出的 ONNX 模型需要显式提供长度
        if len(input_names) > 1:
            processed_signal_length = torch.tensor([processed_signal.shape[2]], device=device).long()
            ort_inputs[input_names[1]] = processed_signal_length.cpu().numpy()
        
        # 4. 运行 ONNX 推理
        logits = session.run(None, ort_inputs)[0] # [B, T, C]
        # 5. 计算语音概率 (C=1 通常是 speech)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=-1)[0, :, 1]
        # 6. 根据阈值判断
        vad_labels = (probabilities > PROCESS_CONFIG.get('nemo_threshold', 0.5))
        # 7. 返回语音帧的比例
        return torch.sum(vad_labels).item() / len(vad_labels) if len(vad_labels) > 0 else 0

# --- 后续主逻辑 ---

# 全局变量，用于在多进程工作器 (worker) 中存储初始化后的模型和配置
VAD_MODELS, PROCESS_CONFIG, LOG_FILE_PATH = {}, {}, None

def setup_logger(log_file):
    """配置主进程的日志记录器。"""
    # 基本配置：同时输出到文件和控制台
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), 
                                  logging.StreamHandler()])
    return logging.getLogger("vad_filter")

def download_models_if_needed(model_names: list, logger):
    """
    在主进程中检查并预下载需要联网的模型 (如 Silero 和 FunASR)，
    以避免多个子进程同时下载。
    """
    logger.info("正在检查并预下载VAD模型...")
    try:
        if 'silero' in model_names:
            logger.info("预加载 Silero...")
            load_silero_vad() # 这将触发 PyTorch Hub 下载
            logger.info("Silero 已就绪。")
        if 'funasr' in model_names:
            from funasr import AutoModel
            logger.info("预加载 FunASR...")
            AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True) # 这将触发 ModelScope 下载
            logger.info("FunASR 已就绪。")
        logger.info("所有需要的模型都已存在于本地缓存中。")
        return True
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        return False

def init_worker(config, log_file_path):
    """
    多进程工作器 (Worker) 的初始化函数。
    
    在每个子进程启动时调用一次，用于：
    1.  初始化全局变量 (PROCESS_CONFIG, LOG_FILE_PATH)。
    2.  为子进程配置独立的日志处理器 (仅写入文件)。
    3.  根据配置，在 *当前子进程* 中加载并实例化 VAD 模型。
    """
    global VAD_MODELS, PROCESS_CONFIG, LOG_FILE_PATH
    PROCESS_CONFIG, LOG_FILE_PATH = config, log_file_path
    
    # 配置子进程日志：只写入文件，避免多进程写入控制台混乱
    logger = logging.getLogger("vad_filter"); logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear() # 清除从主进程继承的 handlers
    fh = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - vad_filter(worker) - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # 在当前进程中实例化 VAD 模型
    for model_name in PROCESS_CONFIG['vad_models']:
        if model_name not in VAD_MODELS:
            try:
                if model_name == 'silero': VAD_MODELS['silero'] = SileroVAD()
                elif model_name == 'funasr': VAD_MODELS['funasr'] = FunASRVAD()
                elif model_name == 'webrtc': VAD_MODELS['webrtc'] = WebRTCVAD(aggressiveness=PROCESS_CONFIG['webrtc_aggressiveness'])
                elif model_name == 'nemo_onnx': VAD_MODELS['nemo_onnx'] = NeMoONNXVAD(onnx_path=PROCESS_CONFIG['nemo_onnx_path'])
            except Exception as e:
                logger.error(f"在子进程中加载模型 {model_name} 失败: {e}")

def process_single_file(input_row):
    """
    工作函数：处理单个音频文件。
    
    参数:
        input_row (list): 来自输入SCP的一行数据。
    
    返回:
        list: input_row 加上新的分析列 [..., vad_status, vad_results_json]
    """
    logger = logging.getLogger("vad_filter")
    # 'final_path' 是上一步 (03_filter) 的输出，位于最后
    audio_path = input_row[-1]
    chunk_id = input_row[0]
    
    try:
        y, sr = sf.read(audio_path, dtype=np.float32)
        vad_results = {} # 存储每个VAD模型的结果
        final_decision = "KEPT" # 默认保留 (纯净噪声)
        
        # 遍历所有在此进程中加载的 VAD 模型
        for model_name, vad_model in VAD_MODELS.items():
            speech_ratio = vad_model.get_speech_ratio(y, sr)
            vad_results[model_name] = f"{speech_ratio:.4f}"
            
            # 核心逻辑：如果 *任何一个* 模型的语音比例超过阈值
            if speech_ratio > PROCESS_CONFIG['speech_ratio_threshold']:
                final_decision = "REJECTED" # 则拒绝此文件 (判定为人声)
        
        # 将详细结果序列化为 JSON 字符串
        results_str = json.dumps(vad_results)
        # 记录日志 (仅写入文件)
        logger.info(f"Processed: {chunk_id:<60} -> Status: {final_decision:<8} | Details: {results_str}")
        return input_row + [final_decision, results_str]
        
    except Exception as e:
        # 捕获音频读取或处理中的异常
        error_str = json.dumps({"error": str(e)})
        logger.error(f"Failed:    {chunk_id:<60} -> Error: {e}")
        return input_row + ["REJECTED", error_str] # 异常文件一律拒绝

def main():
    parser = argparse.ArgumentParser(description='步骤4: 使用多个VAD模型过滤含人声的噪声')
    # --- 输入/输出 ---
    parser.add_argument('--input-scp', required=True, help='输入的 wav.scp (来自步骤3)')
    parser.add_argument('--full-scp', required=True, help='输出的完整分析结果 scp (tsv, 含表头)')
    parser.add_argument('--filtered-scp', required=True, help='输出的筛选后 scp (tsv, 含表头, 仅含KEPT)')
    parser.add_argument('--log-dir', required=True, help='日志保存目录')
    # --- 并行与VAD选择 ---
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--vad-models', nargs='+', required=True, help='要使用的VAD模型列表, e.g., silero funasr')
    parser.add_argument('--speech-ratio-threshold', type=float, default=0.05, help='语音比例阈值 (超过则REJECTED)')
    # --- VAD 模型特定参数 ---
    parser.add_argument('--webrtc-aggressiveness', type=int, default=1, help='(用于webrtc) 激进程度 (0-3)')
    parser.add_argument('--nemo-onnx-path', help='(用于nemo_onnx) 预先导出的ONNX模型文件路径')
    parser.add_argument('--nemo-threshold', type=float, default=0.5, help='(用于nemo_onnx) VAD判决的概率阈值')
    args = parser.parse_args()
    
    # 检查 NeMo ONNX 模型的路径是否提供
    if 'nemo_onnx' in args.vad_models and not args.nemo_onnx_path:
        parser.error("使用 'nemo_onnx' 模型时, --nemo-onnx-path 参数是必需的。")
        
    # 1. 设置主进程日志
    log_file_path = os.path.join(args.log_dir, "04_vad_filtering.log")
    logger = setup_logger(log_file_path)
    logger.info("开始步骤4: VAD人声过滤...");
    logger.info(f"使用模型: {', '.join(args.vad_models)}")
    logger.info(f"人声比例阈值: > {args.speech_ratio_threshold * 100:.1f}%")
    
    # 2. 预下载模型
    if not download_models_if_needed(args.vad_models, logger): 
        logger.error("模型下载失败，退出程序。")
        return
        
    # 3. 准备子进程配置
    config = {
        'vad_models': args.vad_models, 
        'speech_ratio_threshold': args.speech_ratio_threshold, 
        'webrtc_aggressiveness': args.webrtc_aggressiveness, 
        'nemo_onnx_path': args.nemo_onnx_path, 
        'nemo_threshold': args.nemo_threshold
    }
    
    # --- 核心修改 1：使用 Pandas 读取输入 SCP ---
    # Pandas 能正确处理包含复杂字符串 (如 'modified_False') 的 TSV，
    # 而标准的 csv.reader 可能会在默认设置下出错。
    try:
        input_df = pd.read_csv(
            args.input_scp, 
            sep='\t',            # 制表符分隔
            dtype=str,           # 所有列均视为字符串
            keep_default_na=False, # 不将 "NA" 等视为空值
            encoding='utf-8'
        )
        input_header = list(input_df.columns) # 获取表头
        # 将 DataFrame 转换为列表的列表，以便送入多进程
        files_to_process = [list(row) for row in input_df.itertuples(index=False, name=None)]
        
    except pd.errors.EmptyDataError:
        logger.warning(f"输入文件 {args.input_scp} 为空。")
        files_to_process = []
        input_header = [] # 确保 input_header 是一个空列表
    except FileNotFoundError:
        logger.error(f"输入文件未找到: {args.input_scp}")
        return

    logger.info(f"共 {len(files_to_process)} 个音频块待处理。")
    # 定义输出文件的表头
    output_header = input_header + ['vad_status', 'vad_results_json']
    
    # 4. 如果没有文件要处理，创建空的输出文件并退出
    if not files_to_process:
        logger.info("没有文件需要处理。正在创建空的输出文件。")
        empty_df = pd.DataFrame(columns=output_header)
        # 使用 Pandas 写入空文件 (带表头)
        empty_df.to_csv(
            args.full_scp, sep='\t', index=False, encoding='utf-8', 
            quoting=csv.QUOTE_NONE, escapechar=None # 禁用引号和转义
        )
        empty_df.to_csv(
            args.filtered_scp, sep='\t', index=False, encoding='utf-8', 
            quoting=csv.QUOTE_NONE, escapechar=None # 禁用引号和转义
        )
        return

    # 5. 启动多进程池
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers, 
                             initializer=init_worker, # 指定工作器初始化函数
                             initargs=(config, log_file_path)) as executor:
        
        futures = [executor.submit(process_single_file, item) for item in files_to_process]
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="VAD处理进度"):
            all_results.append(future.result())

    logger.info(f"所有 {len(all_results)} 个文件处理完毕，正在写入结果...")

    # --- 核心修改 2：使用 Pandas 写入 full_scp ---
    # 将结果列表转换为 DataFrame
    output_df = pd.DataFrame(all_results, columns=output_header)
    # 使用 Pandas 写入 TSV
    output_df.to_csv(
        args.full_scp,
        sep='\t',
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_NONE, # 关键：不为任何字段添加引号
        escapechar=None         # 关键：不使用任何转义字符
    )

    # --- 核心修改 3：使用 Pandas 过滤并写入 filtered_scp ---
    # 利用 DataFrame 的布尔索引进行高效过滤
    filtered_df = output_df[output_df['vad_status'] == "KEPT"]
    filtered_df.to_csv(
        args.filtered_scp,
        sep='\t',
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_NONE, # 关键：不为任何字段添加引号
        escapechar=None         # 关键：不使用任何转义字符
    )

    # 6. 结束日志
    logger.info("步骤4处理完成！")
    logger.info(f"总输入: {len(output_df)}, 保留 (KEPT): {len(filtered_df)}, 拒绝 (REJECTED): {len(output_df) - len(filtered_df)}")
    logger.info(f"完整分析报告: {args.full_scp}"); 
    logger.info(f"最终纯净噪声清单: {args.filtered_scp}")


if __name__ == "__main__":
    # 确保在所有系统上都使用 'spawn' 启动方法。
    # 这对于 PyTorch、Torchaudio 和其他可能使用 CUDA 的库在多进程中
    # 正确初始化子进程至关重要。
    if os.name != 'posix' or multiprocessing.get_start_method() != 'spawn': 
        multiprocessing.set_start_method('spawn', force=True)
    main()