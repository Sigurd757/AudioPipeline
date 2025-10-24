#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本 05: 声音事件分类 (YAMNet + BEATs 融合)

本脚本是音频处理流程的第五步，也是最后一步。
它承接 `04_filter_by_speech_vad.py` 的输出 (纯净噪声清单)。

核心目标：
使用 YAMNet 和 BEATs 两个模型对每个音频块进行分类，
然后使用一套融合策略 (apply_strategy_refined_no_norm) 来确定
一个最终的分类标签 (fused_label)。

功能特性:
1.  动态加载 `model/BEATs` 目录下的 BEATs 源码。
2.  分别实现 YAMNet (TensorFlow Hub) 和 BEATs (PyTorch) 的分类器类。
3.  YAMNet 返回 (mid, score), BEATs 返回 (display_name, score)。
4.  实现一个融合策略，该策略优先考虑两个模型预测的 "交集"，
    如果没有交集，则在一定置信度下采用最高分，否则标记为 "Uncertain"。
5.  在子进程中加载模型 (YAMNet + BEATs)，并行处理所有音频。
6.  使用 Pandas 库高效、鲁棒地读写 TSV(SCP) 文件，
    特别是为了正确处理输出列中包含的 JSON 字符串 (通过 `quoting=csv.QUOTE_NONE`)。
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import librosa
import json
import logging
import csv # 导入 csv 模块 (主要用于 pandas 的 quoting 常量)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- 动态添加 BEATs 源码路径 ---
# 使得脚本可以从 model/BEATs 子目录中导入 BEATs 模块
script_dir = os.path.dirname(os.path.abspath(__file__))
beats_source_path = os.path.join(script_dir, "model/BEATs") 
if beats_source_path not in sys.path:
    sys.path.append(beats_source_path)
try:
    # 尝试导入 BEATs 模块
    from BEATs import BEATs, BEATsConfig
except ImportError as e:
    print(f"错误：无法在路径 '{beats_source_path}' 中导入 BEATs 模块。"); exit(1)

# --- 屏蔽不必要的日志 ---
# 屏蔽 requests, urllib3 等库的 INFO 日志
logging.basicConfig(level=logging.ERROR) 
# 屏蔽 TensorFlow 加载模型时产生的大量 INFO 和 WARNING 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 

# --- 全局变量 ---
# 用于在多进程工作器中存储模型实例和配置
CLASSIFIERS = {}
PROCESS_CONFIG = {}

# --- 分类器基类 ---
class BaseClassifier:
    """定义分类器模型的抽象基类，统一接口。"""
    def __init__(self, **kwargs):
        self.model = self.load_model(**kwargs)
        
    def load_model(self, **kwargs):
        """[抽象方法] 加载模型。"""
        raise NotImplementedError
        
    def predict(self, y, sr, top_k=3):
        """[抽象方法] 对音频 (y, sr) 进行预测，返回 Top-K 结果。"""
        raise NotImplementedError

# --- YAMNet 实现 (返回 MID) ---
class YAMNetClassifier(BaseClassifier):
    """
    YAMNet 分类器实现。
    使用 TensorFlow Hub 加载模型。
    """
    def load_model(self, model_path: str, **kwargs):
        """
        从 TensorFlow Hub 路径加载 YAMNet 模型。
        
        参数:
            model_path (str): tflite 模型文件路径或 TF Hub URL。
        """
        import tensorflow_hub as hub
        model = hub.load(model_path)
        # YAMNet 模型自带类别映射文件 (CSV)，加载它以获取 index -> mid 的映射
        class_map_path = model.class_map_path().numpy().decode('utf-8')
        yamnet_df = pd.read_csv(class_map_path)
        # 构建一个字典: {index: mid}
        self.idx_to_mid_map = dict(zip(yamnet_df['index'].astype(int), yamnet_df['mid']))
        return model
        
    def predict(self, y, sr, top_k=3):
        """
        对音频进行预测。
        返回:
            list: Top-K 结果，格式为 [(mid, score), ...]
        """
        # YAMNet 模型直接接收 numpy 数组
        scores, _, _ = self.model(y)
        # YAMNet 返回的是逐帧分数，这里取所有帧的平均分
        mean_scores = np.mean(scores, axis=0)
        # 获取分数最高的 K 个索引
        top_indices = np.argsort(mean_scores)[-top_k:][::-1]
        # 转换为 (mid, score) 格式
        return [(self.idx_to_mid_map.get(i, f"未知_{i}"), float(mean_scores[i])) for i in top_indices]

# --- BEATs 实现 (返回 display_name) ---
class BEATsClassifier(BaseClassifier):
    """
    BEATs 分类器实现。
    使用 PyTorch 加载 .pt 检查点。
    """
    def load_model(self, model_path: str, label_csv_path: str, **kwargs):
        """
        从 .pt 检查点文件加载 BEATs 模型。
        
        参数:
            model_path (str): .pt 模型检查点文件路径。
            label_csv_path (str): AudioSet 标签映射文件 (mid -> display_name)。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 从检查点中加载模型配置和状态
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        model.to(self.device); model.eval()
        
        # 加载模型内部的标签字典 (index -> mid)
        self.internal_label_dict = checkpoint.get('label_dict', {})
        # 加载外部提供的标签映射文件 (mid -> display_name)
        self.mid_to_name_map = self._load_mid_to_name_map(label_csv_path)
        return model
        
    def _load_mid_to_name_map(self, label_csv_path):
        """
        辅助函数：加载 AudioSet 标签文件，构建 mid 到 display_name 的映射。
        """
        if not os.path.exists(label_csv_path): return {}
        df = pd.read_csv(label_csv_path, header=0, names=['index', 'mid', 'display_name'])
        return dict(zip(df['mid'], df['display_name']))
        
    def predict(self, y, sr, top_k=3):
        """
        对音频进行预测。
        返回:
            list: Top-K 结果，格式为 [(display_name, score), ...]
        """
        audio_tensor = torch.from_numpy(y).float().unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(audio_tensor.shape, dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            # 提取特征 (对于分类任务，BEATs 返回的是 logits)
            output = self.model.extract_features(audio_tensor, padding_mask=padding_mask)[0]
            
        # BEATs 是多标签分类，使用 Sigmoid 激活
        probs = torch.sigmoid(output).squeeze().cpu()
        top_scores, top_indices = torch.topk(probs, top_k)
        
        results = []
        for i in range(top_k):
            idx = top_indices[i].item(); score = top_scores[i].item()
            # 步骤 1: index -> mid (使用模型内部字典)
            mid = self.internal_label_dict.get(idx, f"未知_{idx}")
            # 步骤 2: mid -> display_name (使用外部标签文件)
            display_name = self.mid_to_name_map.get(mid, mid)
            results.append((display_name, float(score)))
        return results

# --- 融合策略函数 ---
def apply_strategy_refined_no_norm(yamnet_preds, beats_preds, mid_to_name_lookup, uncertainty_threshold=0.5):
    """
    融合 YAMNet 和 BEATs 预测结果的策略函数。
    
    参数:
        yamnet_preds (list): YAMNet 预测结果 [(mid, score), ...]
        beats_preds (list): BEATs 预测结果 [(display_name, score), ...]
        mid_to_name_lookup (dict): MID 到 display_name 的映射字典。
        uncertainty_threshold (float): 不确定性阈值。
        
    返回:
        tuple: (final_label, final_score, strategy)
    """
    
    # 1. 将 YAMNet 的 (mid, score) 转换为 (display_name, score)，以便统一比较
    yamnet_name_preds = {mid_to_name_lookup.get(mid, mid): score for mid, score in yamnet_preds}
    beats_name_preds = dict(beats_preds)
    
    yamnet_labels = set(yamnet_name_preds.keys())
    beats_labels = set(beats_name_preds.keys())
    
    # 2. 计算两个模型 Top-K 结果的交集
    intersection = yamnet_labels & beats_labels
    
    if len(intersection) == 1:
        # 策略1: 唯一交集
        # 如果只有一个共同的预测标签，则采用这个标签
        final_label = list(intersection)[0]
        # 分数取两者中的最大值
        final_score = max(yamnet_name_preds.get(final_label, 0.0), beats_name_preds.get(final_label, 0.0))
        strategy = "Intersection_Unique"
        
    elif len(intersection) > 1:
        # 策略2: 多个交集
        # 如果有多个共同的标签，选择在交集中 "总分"(y_score + b_score) 最高的那个
        best_label = "Unknown"; max_sum_score = -1.0
        for label in intersection:
             y_score = yamnet_name_preds.get(label, 0.0)
             b_score = beats_name_preds.get(label, 0.0)
             current_sum_score = y_score + b_score
             if current_sum_score > max_sum_score: 
                 max_sum_score, best_label = current_sum_score, label
        
        final_label = best_label
        # 分数仍然取两者中的最大值
        final_score = max(yamnet_name_preds.get(final_label, 0.0), beats_name_preds.get(final_label, 0.0))
        strategy = "Intersection_SumScore"
        
    else:
        # 策略3: 没有交集
        # 合并所有预测，取所有标签中分数最高的一个
        all_preds_norm = {}
        for label, score in yamnet_name_preds.items(): all_preds_norm[label] = max(all_preds_norm.get(label, -1.0), score)
        for label, score in beats_name_preds.items(): all_preds_norm[label] = max(all_preds_norm.get(label, -1.0), score)
        
        if not all_preds_norm: 
            return "Error", 0.0, "Prediction_Error"
            
        label_with_max_score = max(all_preds_norm, key=all_preds_norm.get)
        max_overall_score = all_preds_norm[label_with_max_score]
        
        if max_overall_score > uncertainty_threshold:
            # 策略3.1: 最高分超过阈值，采用最高分
            final_label, final_score, strategy = label_with_max_score, max_overall_score, "NoIntersection_MaxScore"
        else:
            # 策略4: 不确定
            # 最高分也低于阈值，标记为 "Uncertain"
            final_label, final_score, strategy = "Uncertain", max_overall_score, "NoIntersection_Uncertain"
            
    return final_label, final_score, strategy

# --- 多进程工作函数 ---

def setup_logger(log_file):
    """配置主进程的日志记录器。"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'), 
                                  logging.StreamHandler()])
    return logging.getLogger("sound_classifier")

def init_worker(config, log_file_path):
    """
    多进程工作器 (Worker) 的初始化函数。
    在每个子进程启动时调用一次。
    """
    global CLASSIFIERS, PROCESS_CONFIG, LOG_FILE_PATH
    CLASSIFIERS = {}; PROCESS_CONFIG = {}; LOG_FILE_PATH = None
    
    PROCESS_CONFIG, LOG_FILE_PATH = config, log_file_path
    
    # 配置子进程日志：只写入文件
    logger = logging.getLogger("sound_classifier"); logger.setLevel(logging.INFO)
    if logger.handlers: logger.handlers.clear()
    fh = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - classifier(worker) - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # 在当前子进程中加载 YAMNet 和 BEATs 模型实例
    print(f"进程 {os.getpid()}: 正在加载模型...")
    CLASSIFIERS['yamnet'] = YAMNetClassifier(model_path=config['yamnet_path'])
    CLASSIFIERS['beats'] = BEATsClassifier(model_path=config['beats_path'], label_csv_path=config['label_csv'])
    # 同时加载 mid -> name 的映射表，供后续使用
    CLASSIFIERS['mid_to_name'] = CLASSIFIERS['beats'].mid_to_name_map
    print(f"进程 {os.getpid()}: 模型加载完毕。")

def process_single_file(input_row):
    """
    工作函数：处理单个音频文件。
    
    参数:
        input_row (list): 来自输入SCP的一行数据。
    
    返回:
        list: input_row 加上新的分析列 
              [..., fused_label, fused_score, fusion_strategy, yamnet_raw_topk, beats_raw_topk]
    """
    logger = logging.getLogger("sound_classifier")
    try:
        # 根据 04_vad_filtering.py 的输出，'final_path' 是第8列 (索引7)
        # 假设 04 的表头是: [chunk_id, ..., status, reason, silence_ratio, final_path (03), vad_status, vad_results_json]
        # 注意：这里假设 'final_path' (来自03) 是我们需要的路径。
        # *** 检查点：根据 04 脚本，'final_path' 在 `input_row[-1]`，即最后一列。
        # *** 检查点：根据 03 脚本，'final_path' 在 `input_row[-1]`。
        # *** 结论：`input_row[-1]` 总是指向最新的有效路径。
        
        # 原始代码使用 `input_row[7]`，这假设了固定的列数。
        # 如果 04 脚本的表头是 `input_header + ['vad_status', 'vad_results_json']`
        # 且 03 的表头是 `input_header(02) + ['status', 'reason', 'silence_ratio', 'final_path']`
        # 且 02 的表头是 `['chunk_id', 'original_audio_id', 'chunk_path', 'duration']`
        # 那么 03 的 'final_path' 索引是 4+4-1 = 7。
        # 那么 04 的 'final_path' (来自03) 索引仍然是 7。
        #
        # 因此，`input_row[7]` 是正确的 (如果遵循了 02->03->04 的流程)。
        
        audio_path = input_row[7] 
        chunk_id = input_row[0]
        
        # 加载音频并重采样到 16kHz
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        top_k = PROCESS_CONFIG.get('top_k', 3)
        
        # 1. YAMNet 预测 (返回 mid)
        yamnet_preds = CLASSIFIERS['yamnet'].predict(y, sr, top_k=top_k)
        # 2. BEATs 预测 (返回 display_name)
        beats_preds = CLASSIFIERS['beats'].predict(y, sr, top_k=top_k)
        
        # 3. 应用融合策略
        fused_label, fused_score, fusion_strategy = apply_strategy_refined_no_norm(
            yamnet_preds, beats_preds, CLASSIFIERS['mid_to_name'], 
            uncertainty_threshold=PROCESS_CONFIG['uncertainty_threshold']
        )
        
        # ==============================================================================
        # 核心修改 1：将 YAMNet 的 MID 转换为显示名称再保存
        # YAMNet 原始预测是 (mid, score)，在保存为 JSON 之前，
        # 我们将其转换为 (display_name, score)，以便报告更具可读性。
        # ==============================================================================
        yamnet_display_preds = [
            (CLASSIFIERS['mid_to_name'].get(mid, mid), score) for mid, score in yamnet_preds
        ]
        
        # 4. 序列化 Top-K 结果为 JSON 字符串
        yamnet_results_json = json.dumps(yamnet_display_preds)
        beats_results_json = json.dumps(beats_preds)

        # 5. 返回追加了所有新列的行
        return input_row + [fused_label, f"{fused_score:.4f}", fusion_strategy, yamnet_results_json, beats_results_json]
        
    except Exception as e:
        # 异常处理
        chunk_id_safe = input_row[0] if len(input_row) > 0 else "UNKNOWN_CHUNK_ID"
        error_str = str(e).replace('\n', ' '); logger.error(f"Failed:    {chunk_id_safe:<60} -> Error: {error_str}")
        return input_row + ["CLASSIFY_ERROR", "0.0", error_str, "[]", "[]"]

def main():
    parser = argparse.ArgumentParser(description="步骤5: 使用YAMNet和BEATs对音频进行融合分类")
    # --- 输入/输出 ---
    parser.add_argument("--input-scp", required=True, help="输入的 wav.scp (来自步骤4)")
    parser.add_argument("--output-scp", required=True, help="输出的最终分类结果 scp (tsv, 含表头)")
    parser.add_argument("--log-dir", required=True, help="日志保存目录")
    # --- 模型与配置 ---
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="并行进程数")
    parser.add_argument("--yamnet-path", required=True, help="YAMNet (TF Hub) 模型路径或URL")
    parser.add_argument("--beats-path", required=True, help="BEATs (.pt) 检查点文件路径")
    parser.add_argument("--label-csv", required=True, help="AudioSet 标签映射文件 (mid -> display_name)")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.5, help="不确定性阈值 (用于融合策略)")
    parser.add_argument("--top-k", type=int, default=3, help="每个模型返回的 Top-K 结果数")
    
    args = parser.parse_args()
    log_file_path = os.path.join(args.log_dir, "05_classify_sounds.log")
    logger = setup_logger(log_file_path)
    logger.info("开始步骤5: 声音事件分类...")
    
    # 将所有命令行参数存入 config 字典，以便传递给子进程
    config = vars(args)

    # --- 核心修改 1：使用 Pandas 读取输入 SCP ---
    # 使用 Pandas 读取 TSV，以鲁棒地处理来自上一步 (04_vad) 可能包含
    # 复杂字符串 (如 JSON) 的列。
    try:
        input_df = pd.read_csv(
            args.input_scp, 
            sep='\t',            # 制表符分隔
            dtype=str,           # 所有列均视为字符串
            keep_default_na=False, # 不将 "NA" 等视为空值
            encoding='utf-8'
        )
        input_header = list(input_df.columns) # 获取表头
        # 将 DataFrame 转换为列表的列表，以兼容 process_single_file 函数
        files_to_process = [list(row) for row in input_df.itertuples(index=False, name=None)]
        
    except pd.errors.EmptyDataError:
        logger.warning(f"输入文件 {args.input_scp} 为空。")
        files_to_process = []
        input_header = [] # 确保 input_header 是一个空列表
    except FileNotFoundError:
        logger.error(f"输入文件未找到: {args.input_scp}")
        return
        
    total_files = len(files_to_process)
    # 定义输出文件的完整表头
    output_header = input_header + ["fused_label", "fused_score", "fusion_strategy", "yamnet_raw_topk", "beats_raw_topk"]

    if total_files == 0:
        logger.info("没有文件需要处理。")
        # 如果没有文件，也创建一个带表头的空输出文件
        pd.DataFrame(columns=output_header).to_csv(
            args.output_scp, 
            sep='\t', 
            index=False, 
            encoding='utf-8', 
            quoting=csv.QUOTE_NONE, # 同样应用 quoting 规则
            escapechar=None
        )
        return

    logger.info(f"共 {total_files} 个文件待分类。使用 {args.workers} 个进程。")

    # 用于收集所有进程返回结果的列表
    results_list = []

    # 启动多进程池，并指定初始化函数
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config, log_file_path)) as executor:
        
        # 提交所有任务
        futures = [executor.submit(process_single_file, row) for row in files_to_process]
        
        # 收集结果
        for future in tqdm(as_completed(futures), total=len(files_to_process), desc="声音分类进度"):
            results_list.append(future.result())

    logger.info(f"所有 {len(results_list)} 个文件处理完毕，正在写入结果...")

    # --- 核心修改 2：使用 Pandas 写入结果 ---
    try:
        # 从结果列表创建 DataFrame
        output_df = pd.DataFrame(results_list, columns=output_header)
        
        # 使用 to_csv 写入，并设置 quoting=csv.QUOTE_NONE
        # 这至关重要，可以防止 Pandas 为包含JSON字符串的列 
        # (如 yamnet_raw_topk) 自动添加引号，确保输出的 SCP 格式纯净。
        output_df.to_csv(
            args.output_scp, 
            sep='\t', 
            index=False, 
            encoding='utf-8',
            quoting=csv.QUOTE_NONE, # 关键：告诉 pandas 不要添加引号
            escapechar=None         # 关键：告诉 pandas 不要转义任何字符
        )
        
        logger.info(f"声音分类完成！结果已保存至: {args.output_scp}")
        
    except Exception as e:
        logger.error(f"写入输出文件失败: {e}")

if __name__ == "__main__":
    # 确保在所有系统上都使用 'spawn' 启动方法。
    # 这对于 PyTorch、TensorFlow 和其他可能使用 CUDA 的库在多进程中
    # 正确初始化子进程至关重要。
    if os.name != 'posix' or multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    main()