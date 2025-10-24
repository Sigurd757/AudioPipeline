其他语言版本：[English](README.md) | [中文](README_zh.md)

# Automated Audio Processing and Classification Pipeline

这是一个自动化的、多阶段的音频处理流水线，用于将大型、异构的音频数据集转换为标准化的、经过严格筛选和分类的数据库。

本流水线通过一个主控 Shell 脚本 (`build_16k_audio_dataset.sh`) 驱动，该脚本按顺序执行 6 个独立的 Python 脚本，以完成从原始音频到最终分类归档的整个流程。该流程具有**断点续传**功能 (通过 `.done` 文件检查)，可以安全地中断和恢复。

## 核心功能

* **标准化与重采样**: 使用 `ffmpeg` 将任意格式的输入音频批量转换为 16-bit PCM、16kHz 单声道的 WAV 文件。
* **音频切分**: 将重采样后的长音频切分为固定时长的音频块（例如 10 秒），并智能处理末尾的短片段（例如保留时长 > 2 秒的片段）。
* **静音处理**:
    * 过滤掉完全静音或静音比例过高的音频块。
    * 自动检测并**缩短**音频块中过长的静音段（例如将超过 2 秒的静音缩短为 0.5-1.5 秒），以提高数据有效性。
* **VAD 人声过滤**: 使用**多个 VAD 模型** (Silero, FunASR, NeMo ONNX, WebRTC) 对音频进行交叉验证，以高置信度筛除所有包含人类语音的片段，确保产出纯净的噪声/音乐数据集。
* **声音事件分类**: 使用 **YAMNet** 和 **BEATs** 两个模型对*所有*音频进行分类，并采用融合策略（优先交集，否则取高分） 为每个音频块生成一个最终的分类标签（如 "Music", "Speech", "Vehicle"）。
* **统计与归档**:
    * 根据 VAD 结果，将所有音频分为三类：`Clean` (纯净), `Suspected` (疑似人声), `Rejected` (确认人声)。
    * 为上述三类分别生成详细的统计报告 (`.json` 和 `.txt`)，按数据集和声音类别（来自步骤5）进行汇总。
    * （可选）将三类文件自动复制并归档到不同的最终目录中，按 `数据集/声音类别/` 结构存放。

## 流水线概览 (Pipeline Overview)

主控脚本 `build_16k_audio_dataset.sh` 按顺序执行以下步骤：

1.  **`01_resample_to_wav.py` (重采样)**
    * **输入**: 一个 `config.json` 文件，定义了原始数据集的路径。
    * **动作**: 使用 `ffmpeg` 将所有音频文件转换为 16kHz WAV。
    * **输出**: `01_resample.scp` (包含所有新 WAV 文件的清单)。

2.  **`02_split_audio_chunks.py` (切分)**
    * **输入**: `01_resample.scp`。
    * **动作**: 将长音频切分为固定时长的音频块。
    * **输出**: `02_split_chunks.scp` (包含所有音频块的清单)。

3.  **`03_filter_by_silence.py` (静音处理)**
    * **输入**: `02_split_chunks.scp`。
    * **动作**: 使用 `librosa` 过滤静音比例过高的片段，并缩短内部的长静音段。
    * **输出**: `03_silence_check_filtered.scp` (处理后保留的音频块清单)。

4.  **`04_filter_by_speech_vad.py` (VAD 人声过滤)**
    * **输入**: `03_silence_check_filtered.scp`。
    * **动作**: 并行运行多个 VAD 模型 (Silero, FunASR, NeMo, WebRTC)，检测人声。
    * **输出**:
        * `04_vad_check_full.scp` (包含所有音频及 VAD 结果的完整报告)。
        * `04_vad_check_filtered.scp` (仅包含 VAD 判定为 "KEPT" 的纯净音频)。

5.  **`05_classification_full.py` (声音分类)**
    * **输入**: `04_vad_check_full.scp` (注意：对 VAD 的*完整报告*进行分类)。
    * **动作**: 使用 YAMNet 和 BEATs 融合模型，为每个音频块预测一个声音标签。
    * **输出**: `05_classification_full.scp` (在 `04` 的清单上追加了分类标签列)。

6.  **`06_statistics_and_copy.py` (统计与归档)**
    * **输入**: `05_classification_full.scp`。
    * **动作**: 读取 VAD 结果和分类标签，进行最终的三分类 (Clean, Suspected, Rejected)，生成统计报告，并（可选）复制文件到最终归档目录。
    * **输出**: `06_clean.scp`, `06_suspected.scp`, `06_rejected.scp` 以及 `06_final_stats.json` / `.txt`。

## 安装与配置 (Setup & Installation)

1.  **克隆仓库**
    ```bash
    git clone https://github.com/Sigurd757/AudioPipeline.git
    cd AudioPipeline
    ```

2.  **安装 FFmpeg**
    此流水线依赖 `ffmpeg` (在 `01_resample_to_wav.py` 中使用)。请确保 `ffmpeg` 已安装并在您的系统 PATH 中。
    ```bash
    # 例如 (Ubuntu/Debian):
    sudo apt update
    sudo apt install ffmpeg
    ```

3.  **安装 Python 依赖**
    ```bash
    pip install -r requirements.txt
    ```


## 使用方法 (Usage)

1.  **创建数据集配置**
    创建一个 JSON 配置文件 (例如 `config/my_datasets.json`)，列出您希望处理的所有原始数据集的名称和路径。
    ```json
    {
      "MyDataset_A": "/path/to/raw/dataset_A",
      "MyDataset_B": "/path/to/another/dataset_B",
      "Freesound_Collection": "/data/Freesound"
    }
    ```

2.  **配置流水线**
    打开 `build_16k_audio_dataset.sh` 脚本，修改顶部的 **"1. 配置参数"** 部分：
    * `dataset_class`: "noise" 或 "music"。
    * `config`: 指向您在步骤 1 中创建的 JSON 配置文件。
    * `workers`: 并行 CPU 进程数。
    * `CUDA_VISIBLE_DEVICES`: 指定用于步骤 5 (分类) 的 GPU ID。
    * 修改所有核心目录 (`target_dir`, `chunks_dir`, `copy_clean_dir` 等) 以匹配您的存储路径。
    * （可选）调整所有阈值参数（如 `chunk_duration`, `silence_db_threshold` 等）。

3.  **运行流水线**
    推荐使用 `nohup` 在后台运行此脚本，并将日志重定向到文件：
    ```bash
    nohup ./build_16k_audio_dataset.sh > pipeline.log 2>&1 &
    ```
    您可以通过 `tail -f pipeline.log` 或查看 `tmp_log/` 目录中各个步骤的详细日志来监控进度。

## 目录结构 (Project Structure)
```
.
├── build_16k_audio_dataset.sh  # (核心) 主控脚本
├── 01_resample_to_wav.py       # 步骤 1: 重采样
├── 02_split_audio_chunks.py    # 步骤 2: 切分
├── 03_filter_by_silence.py     # 步骤 3: 静音处理
├── 04_filter_by_speech_vad.py  # 步骤 4: VAD 过滤
├── 05_classification_full.py   # 步骤 5: 声音分类
├── 06_statistics_and_copy.py   # 步骤 6: 统计与归档
├── requirements.txt            # Python 依赖
│
├── config/                     # 用于存放数据集配置文件
│   ├── 16k_music_dataset1.json
│   └── 16k_noise_dataset1.json
│
└── model/                      # 预训练模型
    ├── BEATs/                  # BEATs
    ├── marblenet/              # NeMo Marblenet VAD
    ├── yamnet/                 # YAMNet
    ├── class_labels_indices.csv
    └── class_labels_indices_norm.csv
```
## 致谢 (Acknowledgements)

本项目在声音事件分类和 VAD 步骤中依赖以下出色的开源项目。感谢他们的贡献：

1.  **Google YAMNet**:
    * **用途**: 声音事件分类。
    * **链接**: <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>

2.  **Microsoft BEATs**:
    * **用途**: 声音事件分类。
    * **链接**: <https://github.com/microsoft/unilm/tree/master/beats>

3.  **NVIDIA NeMo (Marblenet VAD)**:
    * **用途**: 语音活动检测（VAD）。
    * **链接**: <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>

4.  **Silero VAD**:
    * **用途**: 语音活动检测（VAD）。
    * **链接**: <https://github.com/snakers4/silero-vad>

5.  **Alibaba FunASR (FSMN-VAD)**:
    * **用途**: 语音活动检测（VAD）。
    * **链接**: <https://github.com/alibaba-damo-academy/FunASR>

6.  **Google WebRTC VAD**:
    * **用途**: 语音活动检测（VAD）。
    * **链接**: <https://github.com/wiseman/py-webrtcvad>

## 许可证 (License)

本项目采用 **MIT 许可证** 授权。详情请参阅 `LICENSE` 文件。