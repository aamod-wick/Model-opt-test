Here's a comprehensive `CLAUDE.md` file for your Model Optimization testing project:

```markdown
# Model Optimization Test Project

## Project Overview
This project is dedicated to testing and implementing model optimization techniques, specifically focusing on INT8 quantization using NVIDIA ModelOpt and TensorRT. The project explores quantization strategies for ONNX models to improve inference performance while maintaining accuracy.

## Project Structure
```
Model-opt-test/
├── data/                    # Calibration and test datasets
├── data_handler/           # Data processing and handling modules
├── quantize.py            # Main quantization script
├── test1.py               # Testing and validation scripts
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .git/                  # Git version control
```

## Current Working Directory
```
/home/aamod/Model-opt-test
```

## Key Dependencies
Based on the project context, here are the likely requirements:

```txt
# requirements.txt (suggested content)
numpy>=1.24.0
onnx>=1.14.0
onnxruntime>=1.15.0
nvidia-modelopt[onnx]>=0.15.0
tensorrt>=10.0.0
tensorrt-cu12-bindings>=10.0.0
tensorrt-cu12-libs>=10.0.0
pycuda>=2022.1
opencv-python>=4.8.0
matplotlib>=3.7.0
```

## Core Functionality

### Quantization Pipeline
The project implements INT8 quantization using NVIDIA ModelOpt with custom calibration data readers:

```python
# quantize.py structure
- Custom CalibrationDataReader class
- INT8 quantization configuration
- ONNX model loading/saving
- Accuracy validation utilities
```

### Key Features
1. **INT8 Quantization**: Using ModelOpt's PTQ (Post-Training Quantization)
2. **Custom Calibration**: Support for custom calibration data readers
3. **Model Conversion**: ONNX to TensorRT optimization
4. **Performance Testing**: Inference speed and accuracy benchmarks

## Implementation Details

### Calibration Data Reader Template
```python
class CalibrationDataReader:
    def __init__(self, data_path: str, batch_size: int = 1):
        """Initialize with calibration data"""
        pass
    
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        """Return batch of data as {input_name: numpy_array}"""
        pass
```

### Quantization Configuration
```python
moq.quantize(
    onnx_path="path/to/model.onnx",
    quantize_mode="int8",
    calibration_data_reader=custom_reader,
    calibration_method="entropy",  # or "max", "awq_clip", "rtn_dq"
    output_path="quantized_model.onnx"
)
```

## Common Commands

### Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install ModelOpt with ONNX support
pip install -U nvidia-modelopt[onnx]
```

### Running Quantization
```bash
# Run main quantization script
python quantize.py

# Run tests
python test1.py

# Command-line quantization (if using CLI)
python -m modelopt.onnx.quantization \
    --onnx_path model.onnx \
    --quantize_mode int8 \
    --calibration_data_path data/calibration.npz \
    --output_path quantized_model.onnx
```

## Configuration Guidelines

### For INT8 Quantization
- **Calibration Method**: Use `"entropy"` for image models, `"max"` for simpler cases
- **Batch Size**: Typically 1-8 for calibration
- **Data Format**: Calibration data should match model input shape
- **Precision**: INT8 provides 4x model size reduction and up to 2-3x speedup

### Model Compatibility
- **Input Format**: ONNX opset 11+
- **Supported Ops**: Conv, Gemm, MatMul, Add, Mul, etc.
- **TensorRT Version**: 10.0+ recommended

## Testing Strategy

### Validation Steps
1. **Baseline Testing**: Run original FP32 model
2. **Quantization**: Apply INT8 quantization
3. **Accuracy Check**: Compare outputs between FP32 and INT8
4. **Performance Benchmark**: Measure inference latency

### Test Data Structure
```
data/
├── calibration/     # Calibration dataset (100-1000 samples)
├── validation/      # Validation dataset
└── test/           # Final test dataset
```

## Development Workflow

### Current Focus Areas
1. Implementing custom calibration data readers
2. Testing different calibration methods
3. Validating quantization accuracy
4. Optimizing inference performance

### Next Steps
- [ ] Implement accuracy validation metrics
- [ ] Add support for dynamic input shapes
- [ ] Benchmark different calibration methods
- [ ] Create performance comparison reports
- [ ] Add support for FP8 quantization

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Calibration data format mismatch | Ensure data dictionary keys match model input names |
| Memory issues during calibration | Reduce batch size or use fewer calibration samples |
| TensorRT version conflicts | Use compatible versions (10.0.x for now) |
| Missing calibration data | Check data_handler module for generation logic |

### Debug Commands
```bash
# Check TensorRT version
python -c "import tensorrt as trt; print(trt.__version__)"

# Verify ModelOpt installation
python -c "import modelopt.onnx.quantization as moq; print(moq.__file__)"

# Check ONNX model
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"
```

## Project Goals
1. **Primary**: Successfully implement INT8 quantization for target models
2. **Secondary**: Achieve <1% accuracy loss compared to FP32 baseline
3. **Performance**: Achieve 2-3x inference speedup
4. **Production Ready**: Create reusable quantization pipeline

## Notes & Learnings
- **TensorRT 10.0** is stable for INT8 quantization
- Custom calibration readers provide more flexibility than pre-saved data
- The `data_handler` module likely contains data loading and preprocessing logic
- Always validate quantized model outputs before deployment

## Related Resources
- [NVIDIA ModelOpt Documentation](https://docs.nvidia.com/modelopt/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## Version Information
- **Project Started**: [Current Date]
- **Last Updated**: [Current Date]
- **Python Version**: 3.8+ recommended
- **CUDA Version**: 11.8+ recommended

## usage 
# Quick FP8 quantization
from frb_modelopt_quantizer import FRBModelOptQuantizer, QuantizationConfig, QuantizationMode

config = QuantizationConfig(
    mode=QuantizationMode.FP8,
    calibration_method="entropy",
    batch_size=8
)

quantizer = FRBModelOptQuantizer(
    onnx_path="models/frb_model.onnx",
    output_dir="quantized_models",
    config=config
)

quantizer.quantize(
    calibration_h5_files=list(Path("data/calibration").glob("*.h5"))
)