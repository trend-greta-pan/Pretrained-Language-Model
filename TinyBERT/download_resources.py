#!/usr/bin/env python3
"""
Script to download all required resources for TinyBERT training
This includes BERT base models, GLUE datasets, and optionally pre-trained TinyBERT models
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import subprocess

def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    
    return dest_path

def download_bert_base(model_name="bert-base-uncased", save_dir="models/bert-base"):
    """Download BERT base model from Hugging Face"""
    print(f"\n=== Downloading {model_name} ===")
    
    base_url = f"https://huggingface.co/{model_name}/resolve/main"
    save_path = Path(save_dir) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    files_to_download = [
        ("config.json", "config.json"),
        ("pytorch_model.bin", "pytorch_model.bin"),
        ("vocab.txt", "vocab.txt"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("tokenizer.json", "tokenizer.json")
    ]
    
    for remote_file, local_file in files_to_download:
        url = f"{base_url}/{remote_file}"
        dest = save_path / local_file
        
        if dest.exists():
            print(f"  {local_file} already exists, skipping...")
            continue
            
        try:
            download_file(url, dest, f"  Downloading {local_file}")
            print(f"  ✓ Downloaded {local_file}")
        except Exception as e:
            print(f"  ✗ Failed to download {local_file}: {e}")
            # Some files might be optional
            if local_file in ["config.json", "pytorch_model.bin", "vocab.txt"]:
                print(f"    This is a required file. Exiting.")
                sys.exit(1)
    
    print(f"✓ BERT base model saved to: {save_path}")
    return save_path

def download_glue_data(data_dir="data/glue"):
    """Download GLUE benchmark datasets"""
    print("\n=== Downloading GLUE Datasets ===")
    
    # Create a Python script to download GLUE data
    glue_script = '''
import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK_DATA_URLS = {
    'CoLA': 'https://dl.fbaipublicfiles.com/glue/data/CoLA.zip',
    'SST': 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip', 
    'MRPC': 'https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv',
    'QQP': 'https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip',
    'STS': 'https://dl.fbaipublicfiles.com/glue/data/STS-B.zip',
    'MNLI': 'https://dl.fbaipublicfiles.com/glue/data/MNLI.zip',
    'QNLI': 'https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip',
    'RTE': 'https://dl.fbaipublicfiles.com/glue/data/RTE.zip',
    'WNLI': 'https://dl.fbaipublicfiles.com/glue/data/WNLI.zip',
    'diagnostic': 'https://dl.fbaipublicfiles.com/glue/data/AX.tsv'
}

def download_and_extract(task, data_dir):
    print(f"Downloading and extracting {task}...")
    if task == "MRPC":
        print("Processing MRPC...")
        mrpc_dir = os.path.join(data_dir, "MRPC")
        if not os.path.isdir(mrpc_dir):
            os.mkdir(mrpc_dir)
        
        # Download dev IDs
        mrpc_dev_url = TASK_DATA_URLS["MRPC"]
        mrpc_dev_file = os.path.join(mrpc_dir, "dev_ids.tsv")
        urllib.request.urlretrieve(mrpc_dev_url, mrpc_dev_file)
        print(f"Downloaded {mrpc_dev_file}")
        
        # Download additional MRPC data from original sources
        try:
            mrpc_train_url = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
            mrpc_test_url = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
            
            urllib.request.urlretrieve(mrpc_train_url, os.path.join(mrpc_dir, "msr_paraphrase_train.txt"))
            urllib.request.urlretrieve(mrpc_test_url, os.path.join(mrpc_dir, "msr_paraphrase_test.txt"))
            print("Downloaded MRPC train and test files")
        except:
            print("Warning: Could not download additional MRPC files")
            
    elif task == "diagnostic":
        print("Processing diagnostic...")
        ax_file = os.path.join(data_dir, "AX.tsv")
        urllib.request.urlretrieve(TASK_DATA_URLS[task], ax_file)
        print(f"Downloaded {ax_file}")
    else:
        # Download and extract zip file
        data_url = TASK_DATA_URLS[task]
        data_file = os.path.join(data_dir, f"{task}.zip")
        urllib.request.urlretrieve(data_url, data_file)
        
        with zipfile.ZipFile(data_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(data_file)
        print(f"Extracted {task}")

def main(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    for task in TASKS:
        task_dir = os.path.join(data_dir, task if task != 'SST' else 'SST-2')
        if os.path.exists(task_dir) and task != 'MRPC':
            print(f"{task} already exists, skipping...")
            continue
        
        try:
            download_and_extract(task, data_dir)
        except Exception as e:
            print(f"Failed to download {task}: {e}")

if __name__ == "__main__":
    main("''' + data_dir + '''")
'''
    
    # Save and run the GLUE download script
    glue_script_path = Path("download_glue_temp.py")
    with open(glue_script_path, 'w') as f:
        f.write(glue_script)
    
    try:
        subprocess.run([sys.executable, str(glue_script_path)], check=True)
        print(f"✓ GLUE datasets saved to: {data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download GLUE data: {e}")
    finally:
        if glue_script_path.exists():
            glue_script_path.unlink()
    
    return Path(data_dir)

def download_pretrained_tinybert(model_type="4layer", save_dir="models/tinybert"):
    """Download pre-trained TinyBERT models from Google Drive"""
    print(f"\n=== Downloading Pre-trained TinyBERT ({model_type}) ===")
    
    # Google Drive download links (from README)
    models = {
        "4layer": {
            "general_v1": "1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj",
            "general_v2": "1PhI73thKoLU2iliasJmlQXBav3v33-8z",
            "task_specific": "1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg"
        },
        "6layer": {
            "general_v1": "1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x",
            "general_v2": "1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF",
            "task_specific": "1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS"
        }
    }
    
    if model_type not in models:
        print(f"Invalid model type: {model_type}. Choose from: {list(models.keys())}")
        return None
    
    save_path = Path(save_dir) / model_type
    save_path.mkdir(parents=True, exist_ok=True)
    
    def download_from_gdrive(file_id, dest_path):
        """Download file from Google Drive"""
        # Note: For large files, this might require confirmation
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got a virus scan warning page
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(url, stream=True)
                break
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return dest_path
    
    print("Note: Google Drive downloads might require manual intervention for large files.")
    print("If automatic download fails, please download manually from:")
    
    for version, file_id in models[model_type].items():
        print(f"\n  {version}:")
        print(f"    https://drive.google.com/uc?export=download&id={file_id}")
        
        dest_file = save_path / f"tinybert_{model_type}_{version}.zip"
        if dest_file.exists():
            print(f"    Already downloaded, skipping...")
            continue
        
        try:
            download_from_gdrive(file_id, dest_file)
            
            # Extract the zip file
            extract_dir = save_path / version
            with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"    ✓ Extracted to {extract_dir}")
            
            # Optionally remove the zip file
            # dest_file.unlink()
        except Exception as e:
            print(f"    ✗ Failed to download: {e}")
            print(f"    Please download manually from the link above")
    
    return save_path

def download_wikipedia_sample(save_dir="data/corpus"):
    """Download a sample of Wikipedia for training (smaller dataset for testing)"""
    print("\n=== Downloading Wikipedia Sample ===")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # For a real implementation, you would download the full Wikipedia dump
    # This is just a sample for testing
    sample_text = """This is a sample document for training TinyBERT.
Each document should be separated by blank lines.
You can add multiple sentences per document.

This is the second document in the corpus.
It contains different information than the first.
The model will learn from these examples.

Documents can be of varying lengths.
Some might be short, others might be longer.
The important thing is that they are separated by blank lines.
"""
    
    corpus_file = save_path / "sample_corpus.txt"
    with open(corpus_file, 'w') as f:
        f.write(sample_text)
    
    print(f"✓ Sample corpus saved to: {corpus_file}")
    print("\nNote: For actual training, you should download a larger corpus such as:")
    print("  - Wikipedia dump: https://dumps.wikimedia.org/enwiki/")
    print("  - BookCorpus: https://github.com/soskek/bookcorpus")
    print("  - OpenWebText: https://github.com/jcpeterson/openwebtext")
    
    return corpus_file

def create_student_config(teacher_config_path, output_dir="models/student_config", 
                         num_layers=4, hidden_size=312):
    """Create a student model configuration based on teacher config"""
    print("\n=== Creating Student Model Configuration ===")
    
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read teacher config
    with open(teacher_config_path, 'r') as f:
        config = json.load(f)
    
    # Modify for student model
    config['num_hidden_layers'] = num_layers
    config['hidden_size'] = hidden_size
    
    # Adjust intermediate size proportionally
    if hidden_size == 312:
        config['intermediate_size'] = 1200
        config['num_attention_heads'] = 12  # Keep same number of heads
    
    # Save as config.json (what general_distill.py expects)
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Also save with descriptive name for reference
    student_config_file = output_path / f"student_{num_layers}L_{hidden_size}D_config.json"
    with open(student_config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Student config saved to: {config_file}")
    print(f"  Also saved as: {student_config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Download resources for TinyBERT training")
    parser.add_argument("--resource-dir", default="resources", 
                       help="Base directory for all resources")
    parser.add_argument("--download-bert", action="store_true",
                       help="Download BERT base model")
    parser.add_argument("--download-glue", action="store_true",
                       help="Download GLUE datasets")
    parser.add_argument("--download-tinybert", action="store_true",
                       help="Download pre-trained TinyBERT models")
    parser.add_argument("--tinybert-type", default="4layer", choices=["4layer", "6layer"],
                       help="Type of TinyBERT to download")
    parser.add_argument("--download-corpus", action="store_true",
                       help="Download sample corpus for training")
    parser.add_argument("--create-student-config", action="store_true",
                       help="Create student model configuration")
    parser.add_argument("--all", action="store_true",
                       help="Download all resources")
    
    args = parser.parse_args()
    
    resource_dir = Path(args.resource_dir)
    resource_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TinyBERT Resource Downloader")
    print("=" * 60)
    
    # Track what was downloaded
    downloaded = []
    
    # Download BERT base model
    if args.download_bert or args.all:
        bert_path = download_bert_base(
            save_dir=str(resource_dir / "models" / "bert-base")
        )
        downloaded.append(("BERT Base Model", bert_path))
        
        # Create student config if BERT was downloaded
        if args.create_student_config or args.all:
            student_config = create_student_config(
                bert_path / "config.json",
                output_dir=str(resource_dir / "models" / "student_config")
            )
            downloaded.append(("Student Config", student_config))
    
    # Download GLUE datasets
    if args.download_glue or args.all:
        glue_path = download_glue_data(
            data_dir=str(resource_dir / "data" / "glue")
        )
        downloaded.append(("GLUE Datasets", glue_path))
    
    # Download pre-trained TinyBERT
    if args.download_tinybert or args.all:
        tinybert_path = download_pretrained_tinybert(
            model_type=args.tinybert_type,
            save_dir=str(resource_dir / "models" / "tinybert")
        )
        if tinybert_path:
            downloaded.append((f"TinyBERT ({args.tinybert_type})", tinybert_path))
    
    # Download sample corpus
    if args.download_corpus or args.all:
        corpus_path = download_wikipedia_sample(
            save_dir=str(resource_dir / "data" / "corpus")
        )
        downloaded.append(("Sample Corpus", corpus_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    if downloaded:
        print("\n✓ Successfully downloaded:")
        for name, path in downloaded:
            print(f"  - {name}: {path}")
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("\n1. For General Distillation:")
        print(f"   python pregenerate_training_data.py \\")
        print(f"     --train_corpus {resource_dir}/data/corpus/sample_corpus.txt \\")
        print(f"     --bert_model {resource_dir}/models/bert-base/bert-base-uncased \\")
        print(f"     --output_dir {resource_dir}/data/json_corpus \\")
        print(f"     --do_lower_case")
        print(f"\n   python general_distill.py \\")
        print(f"     --pregenerated_data {resource_dir}/data/json_corpus \\")
        print(f"     --teacher_model {resource_dir}/models/bert-base/bert-base-uncased \\")
        print(f"     --student_model {resource_dir}/models/student_config \\")
        print(f"     --output_dir {resource_dir}/models/general_tinybert")
        
        print("\n2. For Task-specific Distillation:")
        print(f"   python task_distill.py \\")
        print(f"     --teacher_model <fine-tuned-bert-path> \\")
        print(f"     --student_model {resource_dir}/models/general_tinybert \\")
        print(f"     --data_dir {resource_dir}/data/glue/<TASK_NAME> \\")
        print(f"     --task_name <TASK_NAME> \\")
        print(f"     --output_dir {resource_dir}/models/task_tinybert")
        
        print("\n3. To use pre-trained TinyBERT (skip general distillation):")
        print(f"   Use models from: {resource_dir}/models/tinybert/")
    else:
        print("\nNo resources were downloaded. Use --help to see available options.")
        print("\nQuick start: python download_resources.py --all")

if __name__ == "__main__":
    main()
