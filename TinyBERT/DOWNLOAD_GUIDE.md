# TinyBERT Resource Download Guide

This guide explains how to use the `download_resources.py` script to download all necessary resources for TinyBERT training.

## Prerequisites

Install required Python packages:
```bash
pip install requests tqdm
```

## Quick Start

Download all resources at once:
```bash
cd TinyBERT
python download_resources.py --all
```

This will download:
- BERT base model from Hugging Face
- GLUE benchmark datasets
- Pre-trained TinyBERT models (4-layer version)
- Sample training corpus
- Create student model configuration

## Selective Downloads

### Download only BERT base model:
```bash
python download_resources.py --download-bert
```

### Download only GLUE datasets:
```bash
python download_resources.py --download-glue
```

### Download only pre-trained TinyBERT:
```bash
# For 4-layer model (default)
python download_resources.py --download-tinybert

# For 6-layer model
python download_resources.py --download-tinybert --tinybert-type 6layer
```

### Download sample corpus:
```bash
python download_resources.py --download-corpus
```

### Create student configuration:
```bash
python download_resources.py --download-bert --create-student-config
```

## Custom Resource Directory

By default, resources are saved to `./resources/`. To use a different directory:
```bash
python download_resources.py --all --resource-dir /path/to/your/resources
```

## Directory Structure

After downloading, your resource directory will look like:
```
resources/
├── models/
│   ├── bert-base/
│   │   └── bert-base-uncased/
│   │       ├── config.json
│   │       ├── pytorch_model.bin
│   │       └── vocab.txt
│   ├── student_config/
│   │   └── student_4L_312D_config.json
│   └── tinybert/
│       └── 4layer/
│           ├── general_v1/
│           ├── general_v2/
│           └── task_specific/
└── data/
    ├── glue/
    │   ├── CoLA/
    │   ├── SST-2/
    │   ├── MRPC/
    │   └── ...
    └── corpus/
        └── sample_corpus.txt
```

## Important Notes

1. **BERT Model Download**: The script downloads from Hugging Face. The `pytorch_model.bin` file is ~440MB.

2. **Google Drive Downloads**: Pre-trained TinyBERT models are hosted on Google Drive. Large files might require manual download if the automatic download fails.

3. **Corpus Data**: The script creates a small sample corpus. For actual training, you need a larger corpus:
   - Wikipedia dump: https://dumps.wikimedia.org/enwiki/
   - BookCorpus: https://github.com/soskek/bookcorpus
   - OpenWebText: https://github.com/jcpeterson/openwebtext

4. **GLUE Data**: The script downloads all GLUE tasks. Each task has train/dev/test splits.

## Manual Download Links

If automatic downloads fail:

### BERT Base Model:
- https://huggingface.co/bert-base-uncased

### Pre-trained TinyBERT:
- 4-layer General v1: https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj
- 4-layer General v2: https://drive.google.com/uc?export=download&id=1PhI73thKoLU2iliasJmlQXBav3v33-8z
- 6-layer General v1: https://drive.google.com/uc?export=download&id=1wXWR00EH
