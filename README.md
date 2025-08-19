# SafetyBERT & SafetyALBERT: Occupational Safety-Specialized Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Domain-adapted BERT and ALBERT models trained on 2.4M occupational safety documents from MSHA, OSHA, NTSB, and other safety organizations.

## Download Models

### Hugging Face Hub

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# SafetyBERT (110M params, highest accuracy)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bert_model = AutoModelForMaskedLM.from_pretrained("adanish91/safetybert")

# SafetyALBERT (12M params, memory efficient)
albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
albert_model = AutoModelForMaskedLM.from_pretrained("adanish91/safetyalbert")
```

**Model Links:**
- [SafetyBERT](https://huggingface.co/adanish91/safetybert)
- [SafetyALBERT](https://huggingface.co/adanish91/safetyalbert)

## Training Data

Download the training dataset (2.4M safety documents, 120MB compressed):

### Option 1: Direct Download
```bash
# Download from Hugging Face Datasets
huggingface-cli download adanish91/safety-training-data all-data-combined.7z --local-dir ./data
```

### Option 2: Python Script
```python
from huggingface_hub import hf_hub_download
import py7zr

# Download compressed dataset
data_file = hf_hub_download(
    repo_id="adanish91/safety-training-data", 
    filename="all-data-combined.7z",
    local_dir="./data"
)

# Extract the data
with py7zr.SevenZipFile(data_file, mode='r') as archive:
    archive.extractall("./data/")
```

**Dataset includes:**
- MSHA, OSHA, NTSB, FRA, IOGP, iChem, Elsevier safety documents
- Pre-processed and cleaned for training
- Ready-to-use CSV format

📁 [**Download Dataset**](https://huggingface.co/datasets/adanish91/safety-training-data)

## Training Your Own Models

### Installation

```bash
pip install torch transformers pandas scikit-learn tqdm matplotlib seaborn numpy
pip install wandb  # optional
```

### Quick Training

```bash
# SafetyBERT
python bert_continual_training.py \
    --data_dir ./data \
    --output_dir ./outputs/safetybert \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-5

# SafetyALBERT  
python albert_continual_training.py \
    --data_dir ./data \
    --output_dir ./outputs/safetyalbert \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --use_mixed_precision
```

### Data Format

Organize CSV files with these columns:
- MSHA: `Narrative`
- OSHA: `Abstract` 
- NTSB: `merged_narrative`
- FRA: `Narrative`
- IOGP: `Narrative`, `What Went Wrong`
- iChem: `Abstract`
- Elsevier: `Abstract`

## Performance

| Model | Parameters | Accuracy Improvement | Use Case |
|-------|------------|---------------------|----------|
| SafetyBERT | 110M | 76.9% vs BERT | Maximum accuracy |
| SafetyALBERT | 12M | 90.3% vs ALBERT | Memory efficient |

Both models significantly outperform their base versions and other models (including Llama 3.1-8B) on safety classification tasks.

## Applications

- Safety document analysis
- Incident classification
- Risk assessment
- Hazard identification

