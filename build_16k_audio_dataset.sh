#!/bin/bash

# ==============================================================================
# 音频处理流水线主控脚本 (Audio Processing Pipeline Master Script)
# ==============================================================================
#
# 描述:
#   此脚本按顺序执行 6 个 Python 脚本，构成一个完整的音频处理流程：
#   1. 重采样 (01_resample_to_wav.py)
#   2. 切分 (02_split_audio_chunks.py)
#   3. 静音过滤 (03_filter_by_silence.py)
#   4. VAD 人声过滤 (04_filter_by_speech_vad.py)
#   5. 声音分类 (05_classification_full.py)
#   6. 统计与归档 (06_statistics_and_copy.py)
#
# 特性:
#   - 支持断点续传 (通过 .done 文件检查)
#   - 详细的日志记录 (每个步骤重定向到 ${tmp_dir} 下的 .log 文件)
#   - 灵活的配置 (所有路径和阈值均在此处定义)
#

# ==============================================================================
# 1. 配置参数 (Configuration Parameters)
# ==============================================================================

# --- 全局配置 (Global Config) ---
export CUDA_VISIBLE_DEVICES=3  # 指定用于 步骤5 (分类) 的GPU
dataset_class="music"           # 数据集类型: "noise" 或 "music"
class_name="Noise"              # 用于构建输出目录 (默认值)
workers=16                      # 步骤 1, 2, 3, 4 的 CPU 并行进程数

# --- 自动路径选择 ---
if [ "${dataset_class}" == "noise" ]; then
    config="./config/16k_noise_dataset2.json" 
    class_name="Noise"
else 
    config="./config/16k_music_dataset1.json" 
    class_name="Music"
fi

# --- 核心目录定义 ---
target_dir="/data2/02_Processed_Data/${class_name}_Processed/16k_Raw/"          # 步骤1: 重采样后音频存放位置
chunks_dir="/data2/02_Processed_Data/${class_name}_Processed/16k_Chunks/"        # 步骤2: 切分后音频块存放位置
silence_processed_dir="/data2/02_Processed_Data/${class_name}_Processed/16k_Silence_Processed/" # 步骤3: 静音缩减后音频存放位置

# --- 派生目录 (日志, SCP) ---
tmp_root="../tmp_log"                                 # 临时文件和日志的根目录
scp_root="/data2/02_Processed_Data/${class_name}_Processed/scp_file" # 所有 .scp 清单文件的根目录
config_filename=$(basename "${config}")
config_basename=$(echo "${config_filename}" | sed 's/\.[^.]*$//') # 根据配置文件名创建唯一的子目录
tmp_dir="${tmp_root}/${config_basename}"              # 当前任务的日志目录
scp_dir="${scp_root}/${config_basename}"              # 当前任务的 SCP 目录

# --- 步骤 1: 重采样 (01_resample_to_wav.py) ---
sample_rate=16000         # 目标采样率

# --- 步骤 2: 切分 (02_split_audio_chunks.py) ---
chunk_duration=10.0       # 音频块目标时长 (秒)
min_last_chunk_duration=2.0 # 最后一个块的最小保留时长 (秒)

# --- 步骤 3: 静音处理 (03_filter_by_silence.py) ---
silence_db_threshold=50.0             # 静音分贝阈值 (e.g., -50dBFS)
total_silence_ratio_threshold=0.9     # 总静音比例阈值 (超过则丢弃)
long_silence_duration_threshold=2.0   # 长静音段阈值 (超过则缩短)
shortened_silence_min_duration=0.5    # 缩短后静音的最小保留时长
shortened_silence_max_duration=1.5    # 缩短后静音的最大保留时长

# --- 步骤 4: VAD 人声过滤 (04_filter_by_speech_vad.py) ---
vad_models=("silero" "funasr" "nemo_onnx") # 使用的 VAD 模型
speech_ratio_threshold=0.05                # 人声比例阈值 (超过则 REJECTED)
nemo_vad_threshold=0.5                     # NeMo VAD 内部阈值
nemo_onnx_path="./model/marblenet/frame_vad_multilingual_marblenet_v2.0.onnx" # NeMo ONNX 模型路径
nemo_model_name="nvidia/frame_vad_multilingual_marblenet_v2.0" # (此变量在脚本中未被使用，但保留)

# --- 步骤 5: 声音分类 (05_classification_full.py) ---
step5_workers=8           # (GPU) 分类任务的工作进程数
yamnet_path="./model/yamnet/" # YAMNet 模型路径
beats_path="./model/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" # BEATs 模型路径
label_csv_path="./model/class_labels_indices_norm.csv" # 标签映射文件 (mid -> display_name)
uncertainty_threshold=0.5 # 融合策略的不确定性阈值

# --- 步骤 6: 统计与归档 (06_statistics_and_copy.py) ---
# (参数已在 步骤4 中定义，此处定义归档目录)
copy_clean_dir="/data5/Final_Pure_${class_name}_Dataset"      # 纯净(Clean)噪声的归档目录
copy_suspected_dir="/data5/Final_Suspected_${class_name}_Dataset" # 疑似(Suspected)噪声的归档目录
copy_rejected_dir="/data5/Final_Rejected_${class_name}_Dataset" # 拒绝(Rejected)噪声的归档目录

# ==============================================================================
# 2. 文件路径定义 (File Path Definitions)
# ==============================================================================
#
# 定义每个步骤的输入/输出 SCP 文件、日志文件和 .done 标记文件
# 这形成了一个依赖链 (e.g., step2_input_scp = step1_scp)

# 步骤 1: 重采样
step1_script="01_resample_to_wav.py"
step1_scp="${scp_dir}/01_resample.scp"
step1_log="${tmp_dir}/01_resample.log"
step1_done="${tmp_dir}/01_resample.done"

# 步骤 2: 切分
step2_script="02_split_audio_chunks.py"
step2_input_scp="${step1_scp}"
step2_output_scp="${scp_dir}/02_split_chunks.scp"
step2_log="${tmp_dir}/02_split_chunks.log"
step2_done="${tmp_dir}/02_split_chunks.done"

# 步骤 3: 静音过滤
step3_script="03_filter_by_silence.py"
step3_input_scp="${step2_output_scp}"
step3_full_scp="${scp_dir}/03_silence_check_full.scp"      # 完整报告
step3_filtered_scp="${scp_dir}/03_silence_check_filtered.scp" # 筛选后 (下一步输入)
step3_log="${tmp_dir}/03_silence_check.log"
step3_done="${tmp_dir}/03_silence_check.done"

# 步骤 4: VAD 人声过滤
step4_script="04_filter_by_speech_vad.py"
step4_input_scp="${step3_filtered_scp}"
step4_full_scp="${scp_dir}/04_vad_check_full.scp"      # 完整报告 (下一步输入)
step4_filtered_scp="${scp_dir}/04_vad_check_filtered.scp" # 筛选后 (纯净噪声)
step4_log="${tmp_dir}/04_vad_check.log"
step4_done="${tmp_dir}/04_vad_check.done"

# 步骤 5: 声音分类
step5_script="05_classification_full.py"
step5_input_scp="${step4_full_scp}" # 注意：对 VAD *完整报告* 进行分类
step5_output_scp="${scp_dir}/05_classification_full.scp"
step5_log="${tmp_dir}/05_classify_sounds.log"
step5_done="${tmp_dir}/05_classify_sounds.done"

# 步骤 6: 统计与归档
step6_script="06_statistics_and_copy.py" # 步骤6 脚本
step6_input_scp="${step5_output_scp}"    # 步骤6 输入 (来自步骤5)
step6_clean_scp="${scp_dir}/06_clean.scp"      # 步骤6 输出 (Clean)
step6_suspected_scp="${scp_dir}/06_suspected.scp" # 步骤6 输出 (Suspected)
step6_rejected_scp="${scp_dir}/06_rejected.scp"   # 步骤6 输出 (Rejected)
step6_stats_json="${scp_dir}/06_final_stats.json"   # 步骤6 输出 (JSON 报告)
step6_stats_txt="${scp_dir}/06_final_stats.txt"     # 步骤6 输出 (TXT 报告)
step6_log="${tmp_dir}/06_filter_and_sort.log"
step6_done="${tmp_dir}/06_filter_and_sort.done"

# ==============================================================================
# 3. 执行流程定义 (Execution Function Definitions)
# ==============================================================================

# --- 3.0 初始化目录 ---
# 确保所有目标目录、SCP 目录和日志目录都存在
mkdir -p "${target_dir}" "${chunks_dir}" "${silence_processed_dir}" "${scp_dir}" "${tmp_dir}"

# --- 步骤 1: 重采样 ---
check_step1() {
    if [ -f "${step1_done}" ]; then
        echo "步骤1: [音频] 重采样已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step1() {
    echo "开始执行 步骤1: [音频] 重采样..."
    python3 "${step1_script}" \
        --target "${target_dir}" \
        --scp "${step1_scp}" \
        --config "${config}" \
        --log-dir "${tmp_dir}" \
        --workers "${workers}" \
        --sr "${sample_rate}" >"${step1_log}" 2>&1
    
    if [ $? -eq 0 ]; then
        touch "${step1_done}"
        echo "步骤1: [音频] 重采样 已成功完成"
    else
        echo "错误: 步骤1失败，查看日志: ${step1_log}"
        exit 1
    fi
}

# --- 步骤 2: 切分 ---
check_step2() {
    if [ -f "${step2_done}" ]; then
        echo "步骤2: [音频] 音频切分已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step2() {
    echo "开始执行 步骤2: [音频] 音频切分..."
    python3 "${step2_script}" \
        --input-scp "${step2_input_scp}" \
        --output-dir "${chunks_dir}" \
        --output-scp "${step2_output_scp}" \
        --chunk-len "${chunk_duration}" \
        --min-last-chunk "${min_last_chunk_duration}" \
        --log-dir "${tmp_dir}" \
        --workers "${workers}" \
        --sr "${sample_rate}" >"${step2_log}" 2>&1
        
    if [ $? -eq 0 ]; then
        touch "${step2_done}"
        echo "步骤2: [音频] 音频切分 已成功完成"
    else
        echo "错误: 步骤2失败，查看日志: ${step2_log}"
        exit 1
    fi
}

# --- 步骤 3: 静音过滤 ---
check_step3() {
    if [ -f "${step3_done}" ]; then
        echo "步骤3: [音频] 静音处理已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step3() {
    echo "开始执行 步骤3: [音频] 静音处理..."
    python3 "${step3_script}" \
        --input-scp "${step3_input_scp}" \
        --output-dir "${silence_processed_dir}" \
        --chunks-dir "${chunks_dir}" \
        --full-scp "${step3_full_scp}" \
        --filtered-scp "${step3_filtered_scp}" \
        --log-dir "${tmp_dir}" \
        --workers "${workers}" \
        --silence-db "${silence_db_threshold}" \
        --ratio-threshold "${total_silence_ratio_threshold}" \
        --long-silence-threshold "${long_silence_duration_threshold}" \
        --shortened-min "${shortened_silence_min_duration}" \
        --shortened-max "${shortened_silence_max_duration}" >"${step3_log}" 2>&1
        
    if [ $? -eq 0 ]; then
        touch "${step3_done}"
        echo "步骤3: [音频] 静音处理 已成功完成"
    else
        echo "错误: 步骤3失败，查看日志: ${step3_log}"
        exit 1
    fi
}

# --- 步骤 4: VAD 人声过滤 ---
check_step4() {
    if [ -f "${step4_done}" ]; then
        echo "步骤4: [音频] VAD人声过滤已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step4() {
    echo "开始执行步骤4: [音频] VAD人声过滤 (模型: ${vad_models[*]})..."
    python3 "${step4_script}" \
        --input-scp "${step4_input_scp}" --full-scp "${step4_full_scp}" --filtered-scp "${step4_filtered_scp}" \
        --log-dir "${tmp_dir}" --workers "${workers}" --vad-models "${vad_models[@]}" \
        --speech-ratio-threshold "${speech_ratio_threshold}" \
        --nemo-onnx-path "${nemo_onnx_path}" --nemo-threshold "${nemo_vad_threshold}" >"${step4_log}" 2>&1
    
    if [ $? -eq 0 ]; then
        touch "${step4_done}"
        echo "步骤4: [音频] VAD人声过滤 已成功完成"
    else
        echo "错误: 步骤4失败，查看日志: ${step4_log}"
        exit 1
    fi
}

# --- 步骤 5: 声音分类 ---
check_step5() {
    if [ -f "${step5_done}" ]; then
        echo "步骤5: [音频] 声音分类已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step5() {
    echo "开始执行步骤5: [音频] 声音事件分类 (使用 ${step5_workers} 个GPU工作进程)..."
    python3 "${step5_script}" \
        --input-scp "${step5_input_scp}" \
        --output-scp "${step5_output_scp}" \
        --log-dir "${tmp_dir}" \
        --workers "${step5_workers}" \
        --yamnet-path "${yamnet_path}" \
        --beats-path "${beats_path}" \
        --label-csv "${label_csv_path}" \
        --uncertainty-threshold "${uncertainty_threshold}" >"${step5_log}" 2>&1
        
    if [ $? -eq 0 ]; then
        touch "${step5_done}"
        echo "步骤5: [音频] 声音分类 已成功完成"
    else
        echo "错误: 步骤5失败，查看日志: ${step5_log}"
        exit 1
    fi
}

# --- 步骤 6: 统计与归档 ---
check_step6() {
    if [ -f "${step6_done}" ]; then
        echo "步骤6: [音频] 筛选与归档已完成, 跳过"
        return 0
    else
        return 1
    fi
}
run_step6() {
    echo "开始执行步骤6: [音频] VAD结果三分类、统计与归档..."
    cmd=(
        python3 "${step6_script}"
        --input-scp "${step6_input_scp}" --vad-models "${vad_models[@]}"
        --speech-ratio-threshold "${speech_ratio_threshold}"
        --output-clean-scp "${step6_clean_scp}" --output-suspected-scp "${step6_suspected_scp}" --output-rejected-scp "${step6_rejected_scp}"
        --stats-json-path "${step6_stats_json}" --stats-txt-path "${step6_stats_txt}"
        --log-dir "${tmp_dir}"
    )
    if [[ -n "${copy_clean_dir}" || -n "${copy_suspected_dir}" || -n "${copy_rejected_dir}" ]]; then
        source_root_dir=$(dirname "${chunks_dir}")
        echo "检测到复制操作，使用源文件根目录: ${source_root_dir}"
        cmd+=(--source-root-dir "${source_root_dir}")
    fi
    if [[ -n "${copy_clean_dir}" ]]; then cmd+=(--copy-clean-to "${copy_clean_dir}"); fi
    if [[ -n "${copy_suspected_dir}" ]]; then cmd+=(--copy-suspected-to "${copy_suspected_dir}"); fi
    if [[ -n "${copy_rejected_dir}" ]]; then cmd+=(--copy-rejected-to "${copy_rejected_dir}"); fi
    
    "${cmd[@]}" >"${step6_log}" 2>&1
    
    if [ $? -eq 0 ]; then
        touch "${step6_done}"
        echo "步骤6: [音频] 筛选与归档 已成功完成"
    else
        echo "错误: 步骤6失败，查看日志: ${step6_log}"
        exit 1
    fi
}

# ==============================================================================
# 4. 主流程 (Main Execution Flow)
# ==============================================================================
echo "===== 开始完整的音频处理流水线 ====="
echo "配置名称: ${config_basename}"
echo "日志目录: ${tmp_dir}"
echo "SCP 目录: ${scp_dir}"

if ! check_step1; then 
    run_step1
fi

if ! check_step2; then 
    run_step2
fi

if ! check_step3; then 
    run_step3
fi

if ! check_step4; then 
    run_step4
fi

if ! check_step5; then 
    run_step5
fi

if ! check_step6; then 
    run_step6
fi

# --- 最终总结 ---
echo ""
echo "===== 所有步骤执行完毕! 您的标准化音频数据集已准备就绪 ====="
echo "纯净音频清单: ${step6_clean_scp}"
echo "疑似人声清单: ${step6_suspected_scp}"
echo "拒绝音频清单: ${step6_rejected_scp}"
echo "最终统计报告 (JSON): ${step6_stats_json}"
echo "最终统计报告 (TXT):  ${step6_stats_txt}"
if [[ -n "${copy_clean_dir}" ]]; then echo "纯净音频已归档至: ${copy_clean_dir}"; fi
if [[ -n "${copy_suspected_dir}" ]]; then echo "疑似人声已归档至: ${copy_suspected_dir}"; fi
if [[ -n "${copy_rejected_dir}" ]]; then echo "拒绝音频已归档至: ${copy_rejected_dir}"; fi


# 推荐的后台运行命令
# nohup ./build_16k_audio_dataset.sh > pipeline.log 2>&1 &
