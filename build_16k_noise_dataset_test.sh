#!/bin/bash

# ==============================================================================
# 1. 配置参数 - 音频处理流程 (完整版 1-6 步)
# ==============================================================================
# ... (所有配置与上一版完全相同) ...
export CUDA_VISIBLE_DEVICES=3
dataset_class="noise"
if [ "${dataset_class}" == "noise" ]; then
    config="./config/16k_noise_dataset4.json" 
    class_name="Noise"
else 
    config="./config/16k_music_dataset1.json" 
    class_name="Music"
fi

target_dir="/data2/tmp/${class_name}_Processed/16k_Raw/"
chunks_dir="/data2/tmp/${class_name}_Processed/16k_Chunks/"
silence_processed_dir="/data2/tmp/${class_name}_Processed/16k_Silence_Processed/"
tmp_root="../tmp_log"
scp_root="/data2/tmp/${class_name}_Processed/scp_file"
config_filename=$(basename "${config}")
config_basename=$(echo "${config_filename}" | sed 's/\.[^.]*$//')
tmp_dir="${tmp_root}/${config_basename}"
scp_dir="${scp_root}/${config_basename}"
workers=16
sample_rate=16000
chunk_duration=10.0; min_last_chunk_duration=2.0
silence_db_threshold=50.0; total_silence_ratio_threshold=0.9; long_silence_duration_threshold=2.0; shortened_silence_min_duration=0.5; shortened_silence_max_duration=1.5
vad_models=("silero" "funasr" "nemo_onnx")
speech_ratio_threshold=0.05
nemo_vad_threshold=0.5
nemo_onnx_path="./model/marblenet/frame_vad_multilingual_marblenet_v2.0.onnx"
nemo_model_name="nvidia/frame_vad_multilingual_marblenet_v2.0"
step5_workers=8
yamnet_path="./model/yamnet/"
beats_path="./model/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
label_csv_path="./model/class_labels_indices_norm.csv"
uncertainty_threshold=0.5
copy_clean_dir="/data5/tmp/Final_Pure_${class_name}_Dataset"
copy_suspected_dir="/data5/tmp/Final_Suspected_${class_name}_Dataset"
copy_rejected_dir="/data5/tmp/Final_Rejected_${class_name}_Dataset"

# ==============================================================================
# 2. 文件路径定义 (更新为6步)
# ==============================================================================
step1_script="01_resample_to_wav.py"; step1_scp="${scp_dir}/01_resample.scp"; step1_log="${tmp_dir}/01_resample.log"; step1_done="${tmp_dir}/01_resample.done"
step2_script="02_split_audio_chunks.py"; step2_input_scp="${step1_scp}"; step2_output_scp="${scp_dir}/02_split_chunks.scp"; step2_log="${tmp_dir}/02_split_chunks.log"; step2_done="${tmp_dir}/02_split_chunks.done"
step3_script="03_filter_by_silence.py"; step3_input_scp="${step2_output_scp}"; step3_full_scp="${scp_dir}/03_silence_check_full.scp"; step3_filtered_scp="${scp_dir}/03_silence_check_filtered.scp"; step3_log="${tmp_dir}/03_silence_check.log"; step3_done="${tmp_dir}/03_silence_check.done"
step4_script="04_filter_by_speech_vad.py"; step4_input_scp="${step3_filtered_scp}"; step4_full_scp="${scp_dir}/04_vad_check_full.scp"; step4_filtered_scp="${scp_dir}/04_vad_check_filtered.scp"; step4_log="${tmp_dir}/04_vad_check.log"; step4_done="${tmp_dir}/04_vad_check.done"
step5_script="05_classification_full.py"; step5_input_scp="${step4_full_scp}"; step5_output_scp="${scp_dir}/05_classification_full.scp"; step5_log="${tmp_dir}/05_classify_sounds.log"; step5_done="${tmp_dir}/05_classify_sounds.done"

# ==============================================================================
# 核心修改：将 step6_script 的名称与您的要求统一
# ==============================================================================
step6_script="06_statistics_and_copy.py" # <--- 命名为 06_...
step6_input_scp="${step5_output_scp}"
step6_clean_scp="${scp_dir}/06_clean.scp"
step6_suspected_scp="${scp_dir}/06_suspected.scp"
step6_rejected_scp="${scp_dir}/06_rejected.scp"
step6_stats_json="${scp_dir}/06_final_stats.json"
step6_stats_txt="${scp_dir}/06_final_stats.txt"
step6_log="${tmp_dir}/06_filter_and_sort.log"
step6_done="${tmp_dir}/06_filter_and_sort.done"

# ==============================================================================
# 3. 执行流程定义
# ==============================================================================
mkdir -p "${target_dir}" "${chunks_dir}" "${silence_processed_dir}" "${scp_dir}" "${tmp_dir}"
# ... (步骤1-4的函数定义保持不变) ...
check_step1() { if [ -f "${step1_done}" ]; then echo "步骤1: [音频] 重采样已完成, 跳过"; return 0; else return 1; fi; }
run_step1() { echo "开始执行步骤1..."; python3 "${step1_script}" --target "${target_dir}" --scp "${step1_scp}" --config "${config}" --log-dir "${tmp_dir}" --workers "${workers}" --sr "${sample_rate}" >"${step1_log}" 2>&1; if [ $? -eq 0 ]; then touch "${step1_done}"; echo "步骤1: 成功"; else echo "错误: 步骤1失败，查看日志: ${step1_log}"; exit 1; fi; }
check_step2() { if [ -f "${step2_done}" ]; then echo "步骤2: [音频] 音频切分已完成, 跳过"; return 0; else return 1; fi; }
run_step2() { echo "开始执行步骤2..."; python3 "${step2_script}" --input-scp "${step2_input_scp}" --output-dir "${chunks_dir}" --output-scp "${step2_output_scp}" --chunk-len "${chunk_duration}" --min-last-chunk "${min_last_chunk_duration}" --log-dir "${tmp_dir}" --workers "${workers}" --sr "${sample_rate}" >"${step2_log}" 2>&1; if [ $? -eq 0 ]; then touch "${step2_done}"; echo "步骤2: 成功"; else echo "错误: 步骤2失败，查看日志: ${step2_log}"; exit 1; fi; }
check_step3() { if [ -f "${step3_done}" ]; then echo "步骤3: [音频] 静音处理已完成, 跳过"; return 0; else return 1; fi; }
run_step3() { echo "开始执行步骤3..."; python3 "${step3_script}" --input-scp "${step3_input_scp}" --output-dir "${silence_processed_dir}" --chunks-dir "${chunks_dir}" --full-scp "${step3_full_scp}" --filtered-scp "${step3_filtered_scp}" --log-dir "${tmp_dir}" --workers "${workers}" --silence-db "${silence_db_threshold}" --ratio-threshold "${total_silence_ratio_threshold}" --long-silence-threshold "${long_silence_duration_threshold}" --shortened-min "${shortened_silence_min_duration}" --shortened-max "${shortened_silence_max_duration}" >"${step3_log}" 2>&1; if [ $? -eq 0 ]; then touch "${step3_done}"; echo "步骤3: 成功"; else echo "错误: 步骤3失败，查看日志: ${step3_log}"; exit 1; fi; }
check_step4() { if [ -f "${step4_done}" ]; then echo "步骤4: [音频] VAD人声过滤已完成, 跳过"; return 0; else return 1; fi; }
run_step4() {
    echo "开始执行步骤4: [音频] VAD人声过滤 (模型: ${vad_models[*]})..."
    python3 "${step4_script}" \
        --input-scp "${step4_input_scp}" --full-scp "${step4_full_scp}" --filtered-scp "${step4_filtered_scp}" \
        --log-dir "${tmp_dir}" --workers "${workers}" --vad-models "${vad_models[@]}" \
        --speech-ratio-threshold "${speech_ratio_threshold}" \
        --nemo-onnx-path "${nemo_onnx_path}" --nemo-threshold "${nemo_vad_threshold}" >"${step4_log}" 2>&1
    if [ $? -eq 0 ]; then touch "${step4_done}"; echo "步骤4: 成功"; else echo "错误: 步骤4失败，查看日志: ${step4_log}"; exit 1; fi;
}
check_step5() { if [ -f "${step5_done}" ]; then echo "步骤5: [音频] 声音分类已完成, 跳过"; return 0; else return 1; fi; }
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
    if [ $? -eq 0 ]; then touch "${step5_done}"; echo "步骤5: 成功"; else echo "错误: 步骤5失败，查看日志: ${step5_log}"; exit 1; fi;
}

check_step6() { if [ -f "${step6_done}" ]; then echo "步骤6: [音频] 筛选与归档已完成, 跳过"; return 0; else return 1; fi; }
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
    if [ $? -eq 0 ]; then touch "${step6_done}"; echo "步骤6: [音频] VAD筛选、统计与归档成功完成。"; else echo "错误: 步骤6失败，查看日志: ${step6_log}"; exit 1; fi
}
# ==============================================================================
# 4. 主流程 (6步全开)
# ==============================================================================
echo "===== 开始完整的音频处理流水线 ====="

if ! check_step1; then run_step1; fi
if ! check_step2; then run_step2; fi
if ! check_step3; then run_step3; fi
if ! check_step4; then run_step4; fi

echo "--- 正在进入第5步: 声音分类 (使用GPU: ${CUDA_VISIBLE_DEVICES:-"All"}) ---"
if ! check_step5; then run_step5; fi

echo "--- 正在进入第6步: 筛选与归档 ---"
if ! check_step6; then run_step6; fi

echo ""
echo "===== 所有步骤执行完毕! 您的标准化音频数据集已准备就绪 ====="
echo "纯净音频清单: ${step6_clean_scp}"
echo "疑似人声清单: ${step6_suspected_scp}"
echo "拒绝音频清单: ${step6_rejected_scp}"
echo "纯净音频统计报告 (JSON): ${step6_stats_json}"
echo "纯净音频统计报告 (TXT):  ${step6_stats_txt}"
if [[ -n "${copy_clean_dir}" ]]; then echo "纯净音频已归档至: ${copy_clean_dir}"; fi
if [[ -n "${copy_suspected_dir}" ]]; then echo "疑似人声已归档至: ${copy_suspected_dir}"; fi
if [[ -n "${copy_rejected_dir}" ]]; then echo "拒绝音频已归档至: ${copy_rejected_dir}"; fi