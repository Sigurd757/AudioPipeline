
Read this in other languages: [English](README.md) | [中文](README_zh.md)

# Automated Audio Processing and Classification Pipeline  

This is an automated, multi-stage audio processing pipeline designed to convert large, heterogeneous audio datasets into a standardized, rigorously filtered, and classified database.  

The pipeline is driven by a core shell script (`build_16k_audio_dataset.sh`), which sequentially executes 6 independent Python scripts to complete the entire workflow—from raw audio files to final classified archiving. It supports **checkpoint resumption** (via `.done` file checks), allowing safe interruption and recovery.  


## Core Features  

- **Standardization & Resampling**: Batch convert audio files of any format to 16-bit PCM, 16kHz mono WAV files using `ffmpeg`.  
- **Audio Splitting**: Split resampled long audio files into fixed-duration chunks (e.g., 10 seconds) and intelligently handle short trailing segments (e.g., retain segments longer than 2 seconds).  
- **Silence Processing**:  
  - Filter out audio chunks that are completely silent or have an excessively high silence ratio.  
  - Automatically detect and **shorten** overly long silent segments within chunks (e.g., reduce silences longer than 2 seconds to 0.5–1.5 seconds) to improve data validity.  
- **VAD Speech Filtering**: Use **multiple VAD models** (Silero, FunASR, NeMo ONNX, WebRTC) for cross-validation to filter out all segments containing human speech with high confidence, ensuring a pure noise/music dataset.  
- **Sound Event Classification**: Classify *all* audio chunks using two models—**YAMNet** and **BEATs**—and adopt a fusion strategy (prioritize intersections, then select high-scoring labels) to assign a final category label (e.g., "Music", "Speech", "Vehicle") to each chunk.  
- **Statistics & Archiving**:  
  - Categorize all audio into three groups based on VAD results: `Clean` (pure non-speech), `Suspected` (potential speech), and `Rejected` (confirmed speech).  
  - Generate detailed statistical reports (`.json` and `.txt`) for the three groups, aggregated by dataset and sound category (from Step 5).  
  - (Optional) Automatically copy and archive files into separate final directories, organized by the structure `Dataset/Sound_Category/`.  


## Pipeline Overview  

The core script `build_16k_audio_dataset.sh` executes the following steps in order:  

### 1. `01_resample_to_wav.py` (Resampling)  
- **Input**: A `config.json` file defining paths to raw datasets.  
- **Action**: Convert all audio files to 16kHz WAV format using `ffmpeg`.  
- **Output**: `01_resample.scp` (a manifest of all new WAV files).  


### 2. `02_split_audio_chunks.py` (Splitting)  
- **Input**: `01_resample.scp`.  
- **Action**: Split long audio files into fixed-duration chunks.  
- **Output**: `02_split_chunks.scp` (a manifest of all audio chunks).  


### 3. `03_filter_by_silence.py` (Silence Filtering)  
- **Input**: `02_split_chunks.scp`.  
- **Action**: Use `librosa` to filter chunks with excessive silence and shorten long silent segments within chunks.  
- **Output**: `03_silence_check_filtered.scp` (a manifest of retained audio chunks after processing).  


### 4. `04_filter_by_speech_vad.py` (VAD Speech Filtering)  
- **Input**: `03_silence_check_filtered.scp`.  
- **Action**: Run multiple VAD models (Silero, FunASR, NeMo, WebRTC) in parallel to detect speech.  
- **Output**:  
  - `04_vad_check_full.scp` (a complete report with VAD results for all audio chunks).  
  - `04_vad_check_filtered.scp` (a manifest of only "KEPT" pure non-speech chunks).  


### 5. `05_classification_full.py` (Sound Classification)  
- **Input**: `04_vad_check_full.scp` (note: classifies the *complete VAD report*).  
- **Action**: Predict a sound label for each chunk using a fused YAMNet + BEATs model.  
- **Output**: `05_classification_full.scp` (appends classification labels to the manifest from Step 4).  


### 6. `06_statistics_and_copy.py` (Statistics & Archiving)  
- **Input**: `05_classification_full.scp`.  
- **Action**: Finalize categorization (Clean/Suspected/Rejected) using VAD results and labels, generate stats, and (optionally) copy files to final directories.  
- **Output**: `06_clean.scp`, `06_suspected.scp`, `06_rejected.scp`, plus `06_final_stats.json` / `.txt`.  


## Setup & Installation  

### 1. Clone the Repository  
```bash
git clone [Your_Repository_URL]
cd [Your_Repository_Name]
```

### 2. Install FFmpeg  
The pipeline depends on `ffmpeg` (used in `01_resample_to_wav.py`). Ensure `ffmpeg` is installed and added to your system’s `PATH`.  
```bash
# Example (Ubuntu/Debian):
sudo apt update
sudo apt install ffmpeg
```

### 3. Install Python Dependencies  
```bash
pip install -r requirements.txt
```


## Usage  

### 1. Create a Dataset Configuration  
Create a JSON config file (e.g., `config/my_datasets.json`) listing names and paths of all raw datasets to process:  
```json
{
  "MyDataset_A": "/path/to/raw/dataset_A",
  "MyDataset_B": "/path/to/another/dataset_B",
  "Freesound_Collection": "/data/Freesound"
}
```

### 2. Configure the Pipeline  
Open the `build_16k_audio_dataset.sh` script and modify the **"1. Configuration Parameters"** section at the top:  
- `dataset_class`: Set to "noise" or "music" (defines the target dataset type).  
- `config`: Path to the JSON config file created in Step 1.  
- `workers`: Number of parallel CPU processes (for multi-process steps).  
- `CUDA_VISIBLE_DEVICES`: Specify GPU ID(s) for Step 5 (classification, e.g., "0" for the first GPU).  
- Update core directories (`target_dir`, `chunks_dir`, `copy_clean_dir`, etc.) to match your storage paths.  
- (Optional) Adjust threshold parameters (e.g., `chunk_duration`, `silence_db_threshold`).  

### 3. Run the Pipeline  
We recommend running the script in the background with `nohup` to avoid interruptions, and redirect logs to a file:  
```bash
nohup ./build_16k_audio_dataset.sh > pipeline.log 2>&1 &
```
- Monitor progress via: `tail -f pipeline.log`  
- View detailed step-specific logs in the `tmp_log/` directory.  


## Project Structure  
```
.
├── build_16k_audio_dataset.sh  # (Core) Main driver script
├── 01_resample_to_wav.py       # Step 1: Audio resampling
├── 02_split_audio_chunks.py    # Step 2: Audio chunking
├── 03_filter_by_silence.py     # Step 3: Silence filtering
├── 04_filter_by_speech_vad.py  # Step 4: VAD speech filtering
├── 05_classification_full.py   # Step 5: Sound classification
├── 06_statistics_and_copy.py   # Step 6: Stats & archiving
├── requirements.txt            # Python dependencies
│
├── config/                     # Dataset config files
│   ├── 16k_music_dataset1.json
│   └── 16k_noise_dataset1.json
│
└── model/                      # Pretrained models
    ├── BEATs/                  # BEATs model
    ├── marblenet/              # NeMo Marblenet VAD model
    ├── yamnet/                 # YAMNet model
    ├── class_labels_indices.csv
    └── class_labels_indices_norm.csv
```


## Acknowledgements  

This project relies on the following outstanding open-source projects for sound classification and VAD. Thank you for their contributions:  

1. **Google YAMNet**  
   - **Purpose**: Sound event classification.  
   - **Link**: <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>  

2. **Microsoft BEATs**  
   - **Purpose**: Sound event classification.  
   - **Link**: <https://github.com/microsoft/unilm/tree/master/beats>  

3. **NVIDIA NeMo (Marblenet VAD)**  
   - **Purpose**: Voice Activity Detection (VAD).  
   - **Link**: <https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0>  

4. **Silero VAD**  
   - **Purpose**: Voice Activity Detection (VAD).  
   - **Link**: <https://github.com/snakers4/silero-vad>  

5. **Alibaba FunASR (FSMN-VAD)**  
   - **Purpose**: Voice Activity Detection (VAD).  
   - **Link**: <https://github.com/alibaba-damo-academy/FunASR>  

6. **Google WebRTC VAD**  
   - **Purpose**: Voice Activity Detection (VAD).  
   - **Link**: <https://github.com/wiseman/py-webrtcvad>  


## License  

This project is licensed under the **MIT License**. See the `LICENSE` file for details.  
