#!/usr/bin/env python3
"""
Download a fine-tuned BERT model for SST-2 from Hugging Face
This will serve as the teacher model for task-specific distillation
"""

import os
import sys
from pathlib import Path

def download_finetuned_bert_sst2():
    """Download fine-tuned BERT for SST-2 from Hugging Face"""
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        print("Installing required packages...")
        os.system("pip install transformers torch")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    
    # Model options (all are BERT fine-tuned on SST-2)
    models = {
        "1": ("textattack/bert-base-uncased-SST-2", "High quality, 93% accuracy"),
        "2": ("howey/bert-base-uncased-sst2", "Alternative option"),
        "3": ("gchhablani/bert-base-cased-finetuned-sst2", "Cased version")
    }
    
    print("\n" + "="*60)
    print("Fine-tuned BERT Model Downloader for SST-2")
    print("="*60)
    print("\nAvailable models:")
    for key, (name, desc) in models.items():
        print(f"  {key}. {name}")
        print(f"     {desc}")
    
    # Default to option 1
    choice = "1"
    model_name, _ = models[choice]
    
    print(f"\n✓ Selected: {model_name}")
    
    # Output directory
    output_dir = Path("resources/models/bert_finetuned_sst2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    
    # Download model and tokenizer
    print("\nDownloading fine-tuned BERT model...")
    print("This may take a few minutes (model size ~440MB)...")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save locally in the format TinyBERT expects
        print(f"\nSaving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Also save in PyTorch format for compatibility
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Fine-tuned BERT downloaded")
        print("="*60)
        
        # Verify files
        print("\nFiles created:")
        for file in output_dir.iterdir():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")
        
        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("\n1. Run intermediate layer distillation:")
        print("   python task_distill.py \\")
        print("     --teacher_model resources/models/bert_finetuned_sst2 \\")
        print("     --student_model resources/models/general_tinybert \\")
        print("     --data_dir resources/data/glue/SST-2 \\")
        print("     --task_name SST-2 \\")
        print("     --output_dir resources/models/tmp_tinybert_sst2 \\")
        print("     --do_lower_case")
        print("\n2. Run prediction layer distillation:")
        print("   python task_distill.py \\")
        print("     --pred_distill \\")
        print("     --teacher_model resources/models/bert_finetuned_sst2 \\")
        print("     --student_model resources/models/tmp_tinybert_sst2 \\")
        print("     --data_dir resources/data/glue/SST-2 \\")
        print("     --task_name SST-2 \\")
        print("     --output_dir resources/models/tinybert_sst2 \\")
        print("     --do_lower_case")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try installing transformers manually: pip install transformers")
        print("3. Try a different model option")
        return False

if __name__ == "__main__":
    success = download_finetuned_bert_sst2()
    sys.exit(0 if success else 1)
