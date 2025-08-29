#!/bin/bash
# Setup script for TinyBERT on AWS SageMaker with GPU support

echo "=========================================="
echo "TinyBERT SageMaker Setup Script"
echo "=========================================="

# Check current PyTorch and CUDA status
echo -e "\n📊 Current Environment Check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Check if CUDA is not available
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo -e "\n⚠️  CUDA is not available with current PyTorch installation!"
    echo "📦 Installing CUDA-enabled PyTorch..."
    
    # Uninstall CPU-only PyTorch
    pip uninstall -y torch torchvision torchaudio
    
    # Install CUDA-enabled PyTorch for CUDA 12.1 (compatible with your CUDA 12.6)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo -e "\n✅ PyTorch with CUDA support installed!"
else
    echo -e "\n✅ CUDA is already available!"
fi

# Verify installation
echo -e "\n📊 Final Environment Check:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Install other requirements
echo -e "\n📦 Installing other dependencies..."
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm boto3 requests

echo -e "\n=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run TinyBERT training with GPU:"
echo "  python task_distill.py \\"
echo "    --teacher_model resources/models/bert_finetuned_sst2 \\"
echo "    --student_model resources/models/general_tinybert \\"
echo "    --data_dir resources/data/glue/SST-2 \\"
echo "    --task_name SST-2 \\"
echo "    --output_dir resources/models/tmp_tinybert_sst2 \\"
echo "    --do_lower_case \\"
echo "    --train_batch_size 256  # Can use large batch with A100s"
echo ""
echo "Note: Do NOT use --no_cuda flag to utilize GPUs!"
