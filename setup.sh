#!/bin/bash

#######################################################################
# Legal OCR Setup Script
# Installs all dependencies and configures the OCR pipeline
#######################################################################

set -e  # Exit on error

echo "======================================================================"
echo "Legal OCR - Installation Script"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Check if running in virtual environment (recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠ Warning: Not running in a virtual environment${NC}"
    echo "It's recommended to use a virtual environment:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # or 'venv\\Scripts\\activate' on Windows"
    echo ""
    echo "Continuing with system Python..."
    echo ""
fi

# Check for GPU (CUDA)
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - will use CPU (slower)${NC}"
    GPU_AVAILABLE=false
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
echo "This may take 5-10 minutes..."
echo ""

pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA or CPU version)
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision
fi

# Install other requirements
echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Download PaddleOCR Portuguese model
echo "Downloading PaddleOCR Portuguese model..."
python -c "from paddleocr import PaddleOCR; print('Initializing PaddleOCR...'); ocr = PaddleOCR(lang='pt', use_angle_cls=True); print('✓ Model downloaded successfully')"

echo -e "${GREEN}✓ PaddleOCR model downloaded${NC}"
echo ""

# Download EasyOCR Portuguese model (optional)
echo "Downloading EasyOCR Portuguese model (fallback - ~400MB)..."
python -c "import easyocr; print('Initializing EasyOCR...'); reader = easyocr.Reader(['pt']); print('✓ Model downloaded successfully')"
echo -e "${GREEN}✓ EasyOCR model downloaded${NC}"
echo ""

# Create logs directory
echo "Creating logs directory..."
mkdir -p logs
touch logs/ocr_processing.log
echo -e "${GREEN}✓ Logs directory created${NC}"
echo ""

# Test installation
echo "Testing installation..."
echo ""

cat > test_ocr.py << 'EOF'
import sys
sys.path.append('.')

try:
    from pipeline_ocr import LegalDocumentOCRPipeline
    print("✓ Pipeline import successful")

    pipeline = LegalDocumentOCRPipeline(use_gpu=False, use_fallback=False)
    print("✓ Pipeline initialization successful")

    print("\nAll tests passed!")
    sys.exit(0)

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_ocr.py
TEST_RESULT=$?
rm test_ocr.py

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "======================================================================"
    echo ""
    echo "Quick start:"
    echo "  python pipeline_ocr.py your_document.pdf"
    echo ""
    echo "For more examples, see:"
    echo "  - SKILL.md - Overview and concepts"
    echo "  - examples.md - Practical usage examples"
    echo "  - reference.md - Technical details"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo -e "${RED}Installation completed with errors${NC}"
    echo "======================================================================"
    echo ""
    echo "Please check the error messages above and try:"
    echo "  pip install -r requirements.txt --upgrade"
    echo ""
    exit 1
fi
