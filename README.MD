# STDiffusion (ICDE 2025)

## Authors
* Zixuan Ma
* Chenfeng Huang

## System Specifications (Tested Environment)
* OS: Ubuntu 22.04
* CPU: Intel(R) Core(TM) i9-11900K (8 cores @ 5.00GHz)
* GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
* CUDA Version: 11.8 or higher

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/mobkageyama/STDiffusion.git
cd STDiffusion
```

### 2. Environment Setup
```bash
# Create and activate conda environment
conda create -n STDiffusion python=3.8
conda activate STDiffusion

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation 
| Dataset | Source | Description |
|---------|---------|-------------|
| ETTh1/ETTh2 | https://github.com/zhouhaoyi/ETDataset | Hourly transformer station data with 8 features over 2 years |
| Energy | https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction | Building energy and environmental data with 28 features at 10-min intervals |
| Exchange_rate | https://github.com/laiguokun/multivariate-time-series-data | Daily exchange rates of 8 countries from 1990-2016 |
| Weather | https://www.bgc-jena.mpg.de/wetter/ | 21 meteorological indicators at 10-min intervals over 1 year |
| Occupancy | https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation | Room occupancy data with 18 sensor features at 30-sec intervals |

## Model Training and Sampling

### Training
Start training using the provided script:
```bash
sh train.sh
```

### Sampling
Generate samples using:
```bash
sh sample.sh
```

### Command Format
```bash
python models/trainer.py \
    --config <configuration_filename> \
    --seed <your_luck_number> \
    --note <custom_label> \
    --action <train or sample>
```

### Command Parameters
- `config`: Configuration file path (e.g., `configs/ETTh1.yaml`)
- `seed`: Random seed for reproducibility
- `note`: Custom label for the experiment
- `action`: Choose between 'train' or 'sample'

### Output Directories
- Training checkpoints: `./STDiffusion/saved_models/`
- Generated samples: `./STDiffusion/generated_datasets/`
