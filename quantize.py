"""
ModelOpt ONNX Quantization - Simple Working Example
This shows the CORRECT interface for calibration data reader
"""

import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
import modelopt.onnx.quantization as moq


# ============================================================================
# PART 1: The CORRECT Calibration Data Reader Interface
# ============================================================================

class CorrectCalibrationDataReader:
    """
    ModelOpt requires a calibration data reader with TWO methods:
    1. get_first() - returns the first batch
    2. get_next()  - returns subsequent batches (or None when done)
    
    Each method must return a dictionary: {input_name: numpy_array}
    The input names MUST match your ONNX model's input names EXACTLY.
    """
    
    def __init__(self, h5_files, input_names, ft_key='data_freq_time', dt_key='data_dm_time'):
        """
        Args:
            h5_files: List of paths to H5 files
            input_names: List of input names from ONNX model
            ft_key: Key for freq-time data in H5 (default: 'data_freq_time')
            dt_key: Key for DM-time data in H5 (default: 'data_dm_time')
        """
        import h5py
        
        self.h5_files = [Path(f) for f in h5_files if Path(f).exists()]
        self.input_names = input_names  # e.g., ['data_freq_time', 'data_dm_time']
        self.ft_key = ft_key
        self.dt_key = dt_key
        self.current_index = 0
        
        print(f"Loaded {len(self.h5_files)} H5 files")
        print(f"Model expects inputs: {self.input_names}")
        
        # Pre-load all data into memory (simplest approach)
        self.all_batches = []
        self._load_all_data()
    
    def _load_all_data(self):
        """Load all H5 data into memory."""
        import h5py
        
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Get data using the correct keys
                    ft_data = f[self.ft_key][:]
                    dt_data = f[self.dt_key][:]
                    
                    # Ensure correct shape (batch, height, width, channels)
                    if ft_data.ndim == 2:
                        ft_data = ft_data.reshape(1, ft_data.shape[0], ft_data.shape[1], 1)
                    if dt_data.ndim == 2:
                        dt_data = dt_data.reshape(1, dt_data.shape[0], dt_data.shape[1], 1)
                    
                    # Convert to float32
                    ft_data = ft_data.astype(np.float32)
                    dt_data = dt_data.astype(np.float32)
                    
                    # Create batch dictionary with correct input names
                    batch = {
                        self.input_names[0]: ft_data,  # 'data_freq_time'
                        self.input_names[1]: dt_data   # 'data_dm_time'
                    }
                    self.all_batches.append(batch)
                    
            except Exception as e:
                print(f"Warning: Could not load {h5_file}: {e}")
        
        print(f"Loaded {len(self.all_batches)} batches for calibration")
    
    def get_first(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the first batch. Required by ModelOpt."""
        self.current_index = 0
        if self.all_batches:
            return self.all_batches[0]
        return None
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the next batch. Required by ModelOpt."""
        self.current_index += 1
        if self.current_index < len(self.all_batches):
            return self.all_batches[self.current_index]
        return None


# ============================================================================
# PART 2: Simple Quantization Function
# ============================================================================

def quantize_model_with_custom_data(
    onnx_path: str,
    output_path: str,
    calibration_h5_files: list,
    quantize_mode: str = "int8",  # "int8", "fp8", or "int4"
    calibration_method: str = "entropy"  # "entropy", "max", "awq_clip"
):
    """
    Quantize an ONNX model using custom H5 calibration data.
    
    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save quantized model
        calibration_h5_files: List of H5 files for calibration
        quantize_mode: "int8", "fp8", or "int4"
        calibration_method: "entropy", "max", "awq_clip", "rtn_dq"
    """
    
    # Step 1: Load the ONNX model to get input names
    print("\n" + "="*60)
    print("STEP 1: Loading ONNX model")
    print("="*60)
    
    model = onnx.load(onnx_path)
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    print(f"Opset version: {model.opset_import[0].version}")
    
    # Step 2: Create calibration data reader
    print("\n" + "="*60)
    print("STEP 2: Creating calibration data reader")
    print("="*60)
    
    calibration_reader = CorrectCalibrationDataReader(
        h5_files=calibration_h5_files,
        input_names=input_names,  # Critical: Use model's input names!
        ft_key='data_freq_time',   # Key in your H5 file
        dt_key='data_dm_time'      # Key in your H5 file
    )
    
    # Step 3: Run quantization
    print("\n" + "="*60)
    print(f"STEP 3: Running {quantize_mode.upper()} quantization")
    print("="*60)
    
    moq.quantize(
        onnx_path=onnx_path,
        quantize_mode=quantize_mode,
        calibration_data_reader=calibration_reader,
        calibration_method=calibration_method,
        output_path=output_path,
    )
    
    print(f"\n✓ Quantization complete! Model saved to: {output_path}")
    
    # Step 4: Verify the quantized model
    print("\n" + "="*60)
    print("STEP 4: Verifying quantized model")
    print("="*60)
    
    verify_model(output_path, calibration_h5_files[:5], input_names)
    
    return output_path


def verify_model(onnx_path, test_h5_files, input_names):
    """Test the quantized model with sample data."""
    import h5py
    
    print("Testing inference with quantized model...")
    
    # Load a sample H5 file
    if not test_h5_files:
        print("No test files provided")
        return
    
    with h5py.File(test_h5_files[0], 'r') as f:
        ft_data = f['data_freq_time'][:]
        dt_data = f['data_dm_time'][:]
    
    # Ensure correct shape
    if ft_data.ndim == 2:
        ft_data = ft_data.reshape(1, ft_data.shape[0], ft_data.shape[1], 1)
    if dt_data.ndim == 2:
        dt_data = dt_data.reshape(1, dt_data.shape[0], dt_data.shape[1], 1)
    
    ft_data = ft_data.astype(np.float32)
    dt_data = dt_data.astype(np.float32)
    
    # Run inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    inputs = {
        input_names[0]: ft_data,
        input_names[1]: dt_data
    }
    
    outputs = session.run(None, inputs)
    print(f"✓ Inference successful! Output shape: {outputs[0].shape}")
    print(f"  Output sample: {outputs[0].flatten()[:5]}")


# ============================================================================
# PART 3: Usage Example
# ============================================================================

def main():
    """Main function to run quantization."""
    
    # Paths (adjust to your actual paths)
    onnx_model = Path("/content/model_a.onnx")
    h5_dir = Path("/content/spotlight-data")
    
    # Find all H5 files
    h5_files = list(h5_dir.glob("**/*.h5"))
    
    if not onnx_model.exists():
        print(f"ERROR: Model not found at {onnx_model}")
        return
    
    if not h5_files:
        print(f"ERROR: No H5 files found in {h5_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files")
    print(f"Model: {onnx_model}")
    
    # Use first 10 files for quick test
    calibration_files = h5_files[:10]
    
    # Run quantization
    quantized_path = quantize_model_with_custom_data(
        onnx_path=str(onnx_model),
        output_path="quantized_models/model_a_int8.onnx",
        calibration_h5_files=calibration_files,
        quantize_mode="int8",      # Try: "int8", "fp8", "int4"
        calibration_method="entropy"  # Try: "entropy", "max", "awq_clip"
    )
    
    print(f"\n{'='*60}")
    print("SUCCESS! Quantization complete.")
    print(f"Quantized model: {quantized_path}")
    print(f"{'='*60}")


# ============================================================================
# PART 4: Understanding ModelOpt's Interface
# ============================================================================

"""
MODELOPT CALIBRATION INTERFACE EXPLANATION:

ModelOpt expects a calibration data reader with this EXACT interface:

class YourDataReader:
    def get_first(self) -> Dict[str, np.ndarray]:
        '''Return the first batch as {input_name: tensor}'''
        pass
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        '''Return subsequent batches, or None when done'''
        pass

Key Points:
1. Dictionary keys MUST match ONNX model's input names exactly
2. Values are numpy arrays (CPU memory, not GPU)
3. Batch dimension should be first (e.g., shape [1, 256, 256, 1])
4. Data type should be float32
5. get_next() returns None when no more data

What ModelOpt does internally:
1. Calls get_first() to get initial batch
2. Calls get_next() repeatedly for calibration
3. Uses these batches to compute quantization scales
4. Applies quantization to the model
5. Saves quantized model
"""


if __name__ == "__main__":
    main()