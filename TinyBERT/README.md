TinyBERT
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 
<br />
<br />
<img src="tinybert_overview.png" width="800" height="210"/>
<br />
<br />

For more details about the techniques of TinyBERT, refer to our paper:

[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26
Add Chinese General_TinyBERT: 2021.7.27

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

Quick Start - Download Required Resources
==========================================
We provide a script to automatically download all required resources (BERT models, datasets, etc.):

```bash
# Download all resources (BERT model, GLUE data, sample corpus, etc.)
python download_resources.py --all

# Or download specific resources:
python download_resources.py --download-bert        # BERT base model
python download_resources.py --download-glue        # GLUE datasets
python download_resources.py --download-tinybert    # Pre-trained TinyBERT
python download_resources.py --download-corpus      # Sample corpus
```

This will create a `resources/` directory with all necessary files. See [DOWNLOAD_GUIDE.md](DOWNLOAD_GUIDE.md) for more details.

**Verify downloads:**
```bash
# Check downloaded files
ls -la resources/models/bert-base/bert-base-uncased/
ls -la resources/data/corpus/
ls -la resources/data/glue/

# Verify BERT model files
du -h resources/models/bert-base/bert-base-uncased/*
```

General Distillation
====================
In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. 

General distillation has three steps: 
1. Download resources (models and corpus)
2. Preprocess the corpus into JSON format
3. Run the transformer distillation

### Step 1: Download Resources
```bash
# Download BERT base model and create student config
python download_resources.py --download-bert --create-student-config

# Download or prepare your corpus (sample provided)
python download_resources.py --download-corpus
```

**Verify Step 1 output:**
```bash
# Check BERT model
ls -la resources/models/bert-base/bert-base-uncased/
# Should contain: config.json, pytorch_model.bin, vocab.txt

# Check student config
cat resources/models/student_config/student_4L_312D_config.json | head -20

# Check corpus
wc -l resources/data/corpus/sample_corpus.txt
```

### Step 2: Preprocess Training Data
Convert raw text corpus to JSON format for efficient training:

```bash
python pregenerate_training_data.py \
    --train_corpus resources/data/corpus/sample_corpus.txt \
    --bert_model resources/models/bert-base/bert-base-uncased \
    --output_dir resources/data/json_corpus \
    --do_lower_case \
    --epochs_to_generate 3
```

This creates JSON files (`epoch_0.json`, `epoch_1.json`, etc.) containing tokenized training examples.

**Note:** For large datasets, you can add `--reduce_memory` flag to reduce memory usage by storing data on disk instead of in memory.

**Verify Step 2 output:**
```bash
# List generated files
ls -la resources/data/json_corpus/
# Should show: epoch_0.json, epoch_0_metrics.json, epoch_1.json, etc.

# Check file sizes
du -h resources/data/json_corpus/*

# View training metrics (number of examples)
cat resources/data/json_corpus/epoch_0_metrics.json

# Count training examples per epoch
wc -l resources/data/json_corpus/epoch_*.json

# Preview first training example (formatted)
head -n 1 resources/data/json_corpus/epoch_0.json | python -m json.tool | head -20
```

**Expected output structure:**
```
resources/data/json_corpus/
├── epoch_0.json           # Training data for epoch 0 (JSONL format)
├── epoch_0_metrics.json   # Metrics for epoch 0 (standard JSON)
├── epoch_1.json           # Training data for epoch 1 (JSONL format)
├── epoch_1_metrics.json   # Metrics for epoch 1 (standard JSON)
├── epoch_2.json           # Training data for epoch 2 (JSONL format)
└── epoch_2_metrics.json   # Metrics for epoch 2 (standard JSON)
```

**Understanding the Output Files:**

The `epoch_X.json` files are in **JSONL format** (JSON Lines) - each line is a separate JSON object representing one training example. You cannot open these directly in a JSON viewer.

**How to view the training data:**
```bash
# View first example (single line)
head -n 1 resources/data/json_corpus/epoch_0.json | python -m json.tool

# View multiple examples
head -n 3 resources/data/json_corpus/epoch_0.json | while read line; do echo "$line" | python -m json.tool; echo "---"; done

# Extract specific fields from first example
head -n 1 resources/data/json_corpus/epoch_0.json | python -c "import json, sys; data=json.loads(sys.stdin.read()); print('Tokens (first 10):', data['tokens'][:10]); print('Is random next:', data['is_random_next'])"

# Convert to readable format (save to file)
python -c "
import json
with open('resources/data/json_corpus/epoch_0.json', 'r') as f:
    for i, line in enumerate(f):
        if i >= 3: break  # Only first 3 examples
        data = json.loads(line)
        print(f'Example {i+1}:')
        print(f'  First 10 tokens: {data[\"tokens\"][:10]}')
        print(f'  Sequence length: {len(data[\"tokens\"])}')
        print(f'  Is random next: {data[\"is_random_next\"]}')
        print()
"
```

The metrics files (`epoch_X_metrics.json`) are standard JSON and can be viewed normally:
```bash
cat resources/data/json_corpus/epoch_0_metrics.json | python -m json.tool
```

### Step 3: Run General Distillation
```bash
python general_distill.py \
    --pregenerated_data resources/data/json_corpus \
    --teacher_model resources/models/bert-base/bert-base-uncased \
    --student_model resources/models/student_config \
    --output_dir resources/models/general_tinybert \
    --do_lower_case \
    --train_batch_size 256
```

**Note:** Add `--reduce_memory` for large datasets. Training may take several hours depending on corpus size and hardware.

**Verify Step 3 output:**
```bash
# Check generated model files
ls -la resources/models/general_tinybert/
# Should contain: pytorch_model.bin, config.json, vocab.txt

# Check model size (should be much smaller than BERT-base)
du -h resources/models/general_tinybert/pytorch_model.bin
# TinyBERT 4-layer: ~14-15MB (vs BERT-base: ~440MB)

# View model config
cat resources/models/general_tinybert/config.json | python -m json.tool | grep -E "num_hidden_layers|hidden_size"

# Check training logs if saved
ls -la resources/models/general_tinybert/log.txt 2>/dev/null && tail -20 resources/models/general_tinybert/log.txt
```

### Skip General Distillation (Use Pre-trained Models)

We provide pre-trained general TinyBERT models. Download them using:

```bash
# Download 4-layer TinyBERT
python download_resources.py --download-tinybert --tinybert-type 4layer

# Download 6-layer TinyBERT
python download_resources.py --download-tinybert --tinybert-type 6layer
```

**Verify pre-trained model download:**
```bash
# Check downloaded models
ls -la resources/models/tinybert/4layer/
# Should contain folders: general_v1/, general_v2/, task_specific/

# Check model files in each version
ls -la resources/models/tinybert/4layer/general_v2/
```

Or download manually:

**1st version (reproduce paper results):**
- [General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 
- [General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

**2nd version (trained with more data):**
- [General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)
- [General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)

**Chinese version:**
- [General_TinyBERT_zh(4layer-312dim)](https://huggingface.co/huawei-noah/TinyBERT_4L_zh/tree/main)
- [General_TinyBERT_zh(6layer-768dim)](https://huggingface.co/huawei-noah/TinyBERT_6L_zh/tree/main)

Data augmentation aims to expand the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

### Step 1: Download GLUE Data
```bash
python download_resources.py --download-glue
```

**Verify GLUE data:**
```bash
# List all GLUE tasks
ls -la resources/data/glue/

# Check specific task data (e.g., SST-2)
ls -la resources/data/glue/SST-2/
# Should contain: train.tsv, dev.tsv, test.tsv

# Count examples in training set
wc -l resources/data/glue/SST-2/train.tsv
```

### Step 2: Run Data Augmentation
```bash
python data_augmentation.py \
    --pretrained_bert_model resources/models/bert-base/bert-base-uncased \
    --glove_embs ${GLOVE_EMB} \
    --glue_dir resources/data/glue \
    --task_name ${TASK_NAME}
```

The augmented dataset `train_aug.tsv` is automatically saved into the corresponding task directory.
TASK_NAME can be one of: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.
Understanding GLUE Data
**GLUE (General Language Understanding Evaluation)** is a benchmark of 9 English language understanding tasks used to evaluate NLP models:

| Task | Full Name | Type | Examples | What It Tests |
|------|-----------|------|----------|---------------|
| **SST-2** | Stanford Sentiment Treebank | Sentiment Analysis | 67k | Positive/negative movie reviews |
| **CoLA** | Corpus of Linguistic Acceptability | Grammar | 8.5k | Is sentence grammatically correct? |
| **MRPC** | Microsoft Research Paraphrase | Paraphrase | 3.7k | Do sentences mean the same? |
| **QQP** | Quora Question Pairs | Duplicate Detection | 364k | Are questions asking same thing? |
| **STS-B** | Semantic Textual Similarity | Similarity (0-5) | 5.7k | How similar are two sentences? |
| **MNLI** | Multi-Genre NLI | Inference | 393k | Entailment/contradiction/neutral |
| **QNLI** | Question NLI | QA Validation | 105k | Does paragraph answer question? |
| **RTE** | Recognizing Textual Entailment | Entailment | 2.5k | Does A entail B? |
| **WNLI** | Winograd NLI | Pronoun Resolution | 634 | What does "it" refer to? |

**Recommended starter task:** SST-2 (sentiment analysis) - simple binary classification with good amount of data.

### Download GLUE Data
GLUE datasets are automatically downloaded when you run:
```bash
python download_resources.py --download-glue  # or --all
```

**Verify GLUE data:**
```bash
# List all GLUE tasks
ls -la resources/data/glue/
# Should show: CoLA/, SST-2/, MRPC/, QQP/, STS-B/, MNLI/, QNLI/, RTE/, WNLI/

# Check specific task (e.g., SST-2)
wc -l resources/data/glue/SST-2/*.tsv
# train.tsv: ~67k lines, dev.tsv: ~872 lines
```

Data Augmentation (Optional)
Data augmentation expands the training set to improve model generalization. **This step is optional** - you can skip it with minimal performance impact.

### Should You Use Data Augmentation?

| Scenario | Use Augmentation? | Impact | Why |
|----------|------------------|--------|-----|
| **Learning/Testing** | ❌ No | -1% accuracy | Simpler, faster, good enough |
| **Small datasets** (RTE, MRPC) | ✅ Yes | +2-3% accuracy | More benefit with less data |
| **Large datasets** (QQP, MNLI) | ❌ No | <1% accuracy | Already enough data |
| **Production** | ✅ Yes | +1-2% accuracy | Every bit helps |

### If You Want Data Augmentation:

**Step 1: Download GloVe embeddings**
```bash
# Option 1: Smaller GloVe (recommended for testing)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d resources/glove/

# Option 2: Larger GloVe (better quality)
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d resources/glove/
```

**Step 2: Run augmentation**
```bash
python data_augmentation.py \
    --pretrained_bert_model resources/models/bert-base/bert-base-uncased \
    --glove_embs resources/glove/glove.6B.300d.txt \
    --glue_dir resources/data/glue \
    --task_name SST-2  # Choose from: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE
```

This creates `train_aug.tsv` with ~2x more training examples.

### To Skip Augmentation:
Simply don't use the `--aug_train` flag in task_distill.py later. The performance difference is typically only 1-2%.
=================
Data augmentation aims to expand the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

### Step 1: Download GLUE Data
```bash
python download_resources.py --download-glue
```

**Verify GLUE data:**
```bash
# List all GLUE tasks
ls -la resources/data/glue/

# Check specific task data (e.g., SST-2)
ls -la resources/data/glue/SST-2/
# Should contain: train.tsv, dev.tsv, test.tsv

# Count examples in training set
wc -l resources/data/glue/SST-2/train.tsv
```

### Step 2: Run Data Augmentation
```bash
python data_augmentation.py \
    --pretrained_bert_model resources/models/bert-base/bert-base-uncased \
    --glove_embs ${GLOVE_EMB} \
    --glue_dir resources/data/glue \
    --task_name ${TASK_NAME}
```

The augmented dataset `train_aug.tsv` is automatically saved into the corresponding task directory.
TASK_NAME can be one of: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

**Verify augmentation output:**
```bash
# Check augmented data (example for SST-2)
ls -la resources/data/glue/SST-2/train_aug.tsv

# Compare original vs augmented data size
wc -l resources/data/glue/SST-2/train.tsv resources/data/glue/SST-2/train_aug.tsv

# Preview augmented examples
head -5 resources/data/glue/SST-2/train_aug.tsv
```

In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

**Prerequisites:** 
- Fine-tuned BERT-base model on your target task
- General TinyBERT (from general distillation or pre-trained)
- Task-specific data (e.g., GLUE)

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.
Task-specific Distillation
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

**Prerequisites:** 
- Fine-tuned BERT-base model on your target task
- General TinyBERT (from general distillation or pre-trained)
- Task-specific data (e.g., GLUE)

### Step 0: Download Fine-tuned BERT Teacher Model

You need a BERT model that's already fine-tuned on your target task. We provide a script to download pre-trained fine-tuned models from Hugging Face:

```bash
# Download fine-tuned BERT for SST-2
python download_finetuned_bert.py
```

This will download a BERT model fine-tuned on SST-2 (93% accuracy) to `resources/models/bert_finetuned_sst2/`.

**Alternative options for getting a fine-tuned teacher:**

1. **Download from Hugging Face manually:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Available fine-tuned models:
# - textattack/bert-base-uncased-SST-2 (recommended, 93% accuracy)
# - howey/bert-base-uncased-sst2
# - gchhablani/bert-base-cased-finetuned-sst2

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

# Save locally
model.save_pretrained("resources/models/bert_finetuned_sst2")
tokenizer.save_pretrained("resources/models/bert_finetuned_sst2")
```

2. **Fine-tune BERT yourself:**
```bash
# Using Hugging Face Transformers
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name sst2 \
    --do_train \
    --data_dir resources/data/glue/SST-2 \
    --output_dir resources/models/bert_finetuned_sst2
```

3. **Use pre-trained models from the TinyBERT team:**
Download the teacher models used in the original paper from their Google Drive links.

**Verify teacher model:**
```bash
ls -la resources/models/bert_finetuned_sst2/
# Should contain: config.json, pytorch_model.bin, vocab.txt
```

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.
==========================
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

**Prerequisites:** 
- Fine-tuned BERT-base model on your target task
- General TinyBERT (from general distillation or pre-trained)
- Task-specific data (e.g., GLUE)

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.

### Step 1: Intermediate Layer Distillation
```bash
python task_distill.py \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model resources/models/general_tinybert \
    --data_dir resources/data/glue/${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --output_dir resources/models/tmp_tinybert \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --aug_train \
    --do_lower_case
```

**Verify intermediate distillation output:**
```bash
# Check model files
ls -la resources/models/tmp_tinybert/
# Should contain: pytorch_model.bin, config.json, vocab.txt

# Check model size
du -h resources/models/tmp_tinybert/pytorch_model.bin

# View training progress (if logged)
ls -la resources/models/tmp_tinybert/*.txt 2>/dev/null && tail -20 resources/models/tmp_tinybert/*.txt
```

### Step 2: Prediction Layer Distillation
```bash
python task_distill.py \
    --pred_distill \
    --teacher_model ${FT_BERT_BASE_DIR} \
    --student_model resources/models/tmp_tinybert \
    --data_dir resources/data/glue/${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --output_dir resources/models/task_tinybert \
    --aug_train \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --eval_step 100 \
    --max_seq_length 128 \
    --train_batch_size 32
```

**Verify final model output:**
```bash
# Check final model files
ls -la resources/models/task_tinybert/
# Should contain: pytorch_model.bin, config.json, vocab.txt, eval_results.txt

# Check evaluation results
cat resources/models/task_tinybert/eval_results.txt

# Compare model sizes
du -h resources/models/task_tinybert/pytorch_model.bin ${FT_BERT_BASE_DIR}/pytorch_model.bin
```

### Pre-trained Task-specific Models

We also provide distilled TinyBERT for all GLUE tasks:
- [TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 
- [TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)

Evaluation
==========
Evaluate your TinyBERT model:

```bash
python task_distill.py \
    --do_eval \
    --student_model resources/models/task_tinybert \
    --data_dir resources/data/glue/${TASK_NAME} \
    --task_name ${TASK_NAME} \
    --output_dir resources/eval_results \
    --do_lower_case \
    --eval_batch_size 32 \
    --max_seq_length 128
```

**Check evaluation results:**
```bash
# View evaluation metrics
cat resources/eval_results/eval_results.txt

# Check detailed results if available
ls -la resources/eval_results/
cat resources/eval_results/*.txt
```

Here's a complete example for training TinyBERT on SST-2 (sentiment analysis):

```bash
# 1. Download all resources
python download_resources.py --all

# Verify downloads
ls -la resources/models/bert-base/bert-base-uncased/
ls -la resources/data/glue/SST-2/

# 2. Preprocess corpus for general distillation
python pregenerate_training_data.py \
    --train_corpus resources/data/corpus/sample_corpus.txt \
    --bert_model resources/models/bert-base/bert-base-uncased \
    --output_dir resources/data/json_corpus \
    --do_lower_case

# Verify preprocessing
wc -l resources/data/json_corpus/epoch_*.json
cat resources/data/json_corpus/epoch_0_metrics.json

# 3. General distillation (or skip by using pre-trained)
python general_distill.py \
    --pregenerated_data resources/data/json_corpus \
    --teacher_model resources/models/bert-base/bert-base-uncased \
    --student_model resources/models/student_config \
    --output_dir resources/models/general_tinybert

# Verify general model
ls -la resources/models/general_tinybert/
du -h resources/models/general_tinybert/pytorch_model.bin

# 4. Fine-tune BERT on SST-2 (teacher preparation)
# ... (use your preferred method to fine-tune BERT on SST-2)
# Or download a pre-trained fine-tuned model

# 5. Task-specific distillation - Intermediate layer
python task_distill.py \
    --teacher_model path/to/fine-tuned-bert-sst2 \
    --student_model resources/models/general_tinybert \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tmp_tinybert_sst2 \
    --do_lower_case

# Verify intermediate model
ls -la resources/models/tmp_tinybert_sst2/

# 6. Task-specific distillation - Prediction layer
python task_distill.py \
    --pred_distill \
    --teacher_model path/to/fine-tuned-bert-sst2 \
    --student_model resources/models/tmp_tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tinybert_sst2 \
    --do_lower_case

# Verify final model
ls -la resources/models/tinybert_sst2/
cat resources/models/tinybert_sst2/eval_results.txt

# 7. Evaluate
python task_distill.py \
    --do_eval \
    --student_model resources/models/tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/eval_results \
    --do_lower_case

# Check final evaluation
cat resources/eval_results/eval_results.txt
```
Complete Step-by-Step Workflow
Here's a complete step-by-step guide for training TinyBERT on SST-2 (sentiment analysis):

## Step 1: Download All Resources
```bash
python download_resources.py --all
```
**Verify:** Check that all resources are downloaded:
```bash
ls -la resources/models/bert-base/bert-base-uncased/  # BERT model files
ls -la resources/data/glue/SST-2/                     # SST-2 dataset
ls -la resources/data/corpus/                          # Sample corpus
```
**Expected:** BERT model files, SST-2 dataset with train/dev/test.tsv files

## Step 2: Preprocess Corpus for General Distillation
```bash
python pregenerate_training_data.py \
    --train_corpus resources/data/corpus/sample_corpus.txt \
    --bert_model resources/models/bert-base/bert-base-uncased \
    --output_dir resources/data/json_corpus \
    --do_lower_case \
    --epochs_to_generate 3
```
**Verify:** Check preprocessed data:
```bash
ls -la resources/data/json_corpus/
cat resources/data/json_corpus/epoch_0_metrics.json
```
**Expected:** epoch_0.json, epoch_1.json, epoch_2.json files with metrics

## Step 3: Run General Distillation
```bash
python general_distill.py \
    --pregenerated_data resources/data/json_corpus \
    --teacher_model resources/models/bert-base/bert-base-uncased \
    --student_model resources/models/student_config \
    --output_dir resources/models/general_tinybert \
    --do_lower_case \
    --train_batch_size 256
```
**Verify:** Check general TinyBERT model:
```bash
ls -la resources/models/general_tinybert/
du -h resources/models/general_tinybert/*.bin
```
**Expected:** Model ~57MB (7.5x smaller than BERT's 440MB)
**Time:** ~1-2 hours on GPU, longer on CPU

## Step 4: Download Fine-tuned BERT Teacher
```bash
python download_finetuned_bert.py
```
**Verify:** Check teacher model:
```bash
ls -la resources/models/bert_finetuned_sst2/
```
**Expected:** config.json, pytorch_model.bin, vocab.txt files

## Step 5: Intermediate Layer Distillation
```bash
python task_distill.py \
    --teacher_model resources/models/bert_finetuned_sst2 \
    --student_model resources/models/general_tinybert \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tmp_tinybert_sst2 \
    --do_lower_case \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --num_train_epochs 10
```
**GPU/CPU Usage:**
- **GPU (default):** Automatically uses GPU if available
- **CPU only:** Add `--no_cuda` flag (will be 10x slower)
- **Note:** If you encounter DataParallel errors on CPU, run `python fix_task_distill.py` first
**What happens:** 
- Trains TinyBERT's intermediate layers
- Learns from teacher's hidden states and attention
- Processes 67,349 SST-2 training examples

**Verify:** Check intermediate model:
```bash
ls -la resources/models/tmp_tinybert_sst2/
```
**Expected:** pytorch_model.bin, config.json, vocab.txt
**Time:** ~30-60 minutes on GPU

## Step 6: Prediction Layer Distillation
```bash
python task_distill.py \
    --pred_distill \
    --teacher_model resources/models/bert_finetuned_sst2 \
    --student_model resources/models/tmp_tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tinybert_sst2 \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 3
```
**What happens:**
- Fine-tunes the prediction layer
- Optimizes for SST-2 classification task
- Final model ready for deployment

**Verify:** Check final model:
```bash
ls -la resources/models/tinybert_sst2/
```
**Expected:** Final TinyBERT model files
**Time:** ~10-20 minutes on GPU

## Step 7: Evaluate Final Model
```bash
python task_distill.py \
    --do_eval \
    --student_model resources/models/tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/eval_results \
    --do_lower_case
```
**Verify:** Check results:
```bash
cat resources/eval_results/eval_results.txt
```
**Expected Performance:**
- TinyBERT: ~90% accuracy
- BERT-base: ~93% accuracy
- **Result: 96.8% of BERT performance at 7.5x smaller size!**

## Summary
| Step | Purpose | Time (GPU) | Output |
|------|---------|------------|--------|
| 1 | Download resources | 5 min | BERT, GLUE, corpus |
| 2 | Preprocess corpus | 5 min | JSON training data |
| 3 | General distillation | 1-2 hrs | General TinyBERT |
| 4 | Get teacher model | 2 min | Fine-tuned BERT |
| 5 | Intermediate distill | 30-60 min | Intermediate model |
| 6 | Prediction distill | 10-20 min | Final TinyBERT |
| 7 | Evaluation | 2 min | Performance metrics |

**Total: ~3-4 hours on GPU** (multiply by 10x for CPU)

## Tips for Success
- **GPU Recommended**: Training is 10x faster on GPU
- **Batch Size**: Reduce if out of memory (16 or 8)
- **Skip Steps**: Use pre-trained models to skip general distillation
- **Data Augmentation**: Optional, adds ~1-2% accuracy
=========================
Here's a complete example for training TinyBERT on SST-2 (sentiment analysis):

```bash
# 1. Download all resources
python download_resources.py --all

# Verify downloads
ls -la resources/models/bert-base/bert-base-uncased/
ls -la resources/data/glue/SST-2/

# 2. Preprocess corpus for general distillation
python pregenerate_training_data.py \
    --train_corpus resources/data/corpus/sample_corpus.txt \
    --bert_model resources/models/bert-base/bert-base-uncased \
    --output_dir resources/data/json_corpus \
    --do_lower_case

# Verify preprocessing
wc -l resources/data/json_corpus/epoch_*.json
cat resources/data/json_corpus/epoch_0_metrics.json

# 3. General distillation (or skip by using pre-trained)
python general_distill.py \
    --pregenerated_data resources/data/json_corpus \
    --teacher_model resources/models/bert-base/bert-base-uncased \
    --student_model resources/models/student_config \
    --output_dir resources/models/general_tinybert

# Verify general model
ls -la resources/models/general_tinybert/
du -h resources/models/general_tinybert/pytorch_model.bin

# 4. Fine-tune BERT on SST-2 (teacher preparation)
# ... (use your preferred method to fine-tune BERT on SST-2)
# Or download a pre-trained fine-tuned model

# 5. Task-specific distillation - Intermediate layer
python task_distill.py \
    --teacher_model path/to/fine-tuned-bert-sst2 \
    --student_model resources/models/general_tinybert \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tmp_tinybert_sst2 \
    --do_lower_case

# Verify intermediate model
ls -la resources/models/tmp_tinybert_sst2/

# 6. Task-specific distillation - Prediction layer
python task_distill.py \
    --pred_distill \
    --teacher_model path/to/fine-tuned-bert-sst2 \
    --student_model resources/models/tmp_tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/models/tinybert_sst2 \
    --do_lower_case

# Verify final model
ls -la resources/models/tinybert_sst2/
cat resources/models/tinybert_sst2/eval_results.txt

# 7. Evaluate
python task_distill.py \
    --do_eval \
    --student_model resources/models/tinybert_sst2 \
    --data_dir resources/data/glue/SST-2 \
    --task_name SST-2 \
    --output_dir resources/eval_results \
    --do_lower_case

# Check final evaluation
cat resources/eval_results/eval_results.txt
```

Important Notes
1. **Corpus Size**: The sample corpus is small. For real training, use larger corpora:
   - Wikipedia dump: https://dumps.wikimedia.org/enwiki/
   - BookCorpus: https://github.com/soskek/bookcorpus
   - OpenWebText: https://github.com/jcpeterson/openwebtext

2. **Teacher Models**: You need fine-tuned BERT models for task-specific distillation. Train these separately or use existing ones.

3. **Resource Requirements**: Full training requires significant computational resources (GPU recommended).

4. **Output Locations Summary**:
   ```
   resources/
   ├── models/
   │   ├── bert-base/           # Downloaded BERT base model
   │   ├── student_config/      # Student model configuration
   │   ├── general_tinybert/    # General distilled model
   │   ├── tmp_tinybert/        # Intermediate task-specific model
   │   ├── task_tinybert/       # Final task-specific model
   │   └── tinybert/            # Pre-trained models (if downloaded)
   ├── data/
   │   ├── corpus/              # Raw text corpus
   │   ├── json_corpus/         # Preprocessed training data
   │   └── glue/                # GLUE benchmark datasets
   └── eval_results/            # Evaluation outputs
   ```

To Dos
* Evaluate TinyBERT on Chinese tasks.
* Tiny*: use NEZHA or ALBERT as the teacher in TinyBERT learning.
* Release better general TinyBERTs.
