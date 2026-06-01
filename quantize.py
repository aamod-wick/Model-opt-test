"""
ModelOpt Quantization for FRB Detector (Radio Transient Model)
Supports FP8, INT8, and INT4 quantization modes
Reads H5 files with DM-time and Freq-time data
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Union
import modelopt.onnx.quantization as moq
import onnx
import onnxruntime as ort
from dataclasses import dataclass
from enum import Enum
import time
import json


class QuantizationMode(Enum):
    """Supported quantization modes"""
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """Configuration for ModelOpt quantization"""
    mode: QuantizationMode = QuantizationMode.FP8
    calibration_method: str = "entropy"  # "max", "entropy", "awq_clip", "rtn_dq"
    batch_size: int = 8
    use_cache: bool = True
    op_types_to_quantize: Optional[List[str]] = None
    op_types_to_exclude: Optional[List[str]] = None
    
    def __post_init__(self):
        # Default quantization targets for FRB detector
        if self.op_types_to_quantize is None:
            self.op_types_to_quantize = ["Conv", "Gemm", "MatMul", "Add", "Mul"]
        
        # Exclude certain ops for better accuracy
        if self.op_types_to_exclude is None:
            self.op_types_to_exclude = ["Softmax", "LayerNormalization", "ReduceMean"]


class FRBCalibrationDataReader:
    """
    Calibration data reader for FRB detector model.
    Reads H5 files containing DM-time and Freq-time data.
    
    H5 file structure expectation:
        - 'ft' or 'freq_time': Freq-time data shape (B, 256, 256, 1) or (256, 256, 1)
        - 'dt' or 'dm_time': DM-time data shape (B, 256, 256, 1) or (256, 256, 1)
    """
    
    def __init__(
        self,
        h5_files: List[Union[str, Path]],
        batch_size: int = 8,
        input_names: List[str] = None,
        ft_key: str = "ft",
        dt_key: str = "dt",
        normalize: bool = True,
        shuffle: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Initialize FRB calibration data reader.
        
        Args:
            h5_files: List of H5 file paths containing calibration data
            batch_size: Batch size for calibration (default: 8)
            input_names: ONNX input names ['freq_time', 'dm_time'] or custom
            ft_key: Key for freq-time data in H5 file (default: "ft")
            dt_key: Key for DM-time data in H5 file (default: "dt")
            normalize: Whether to normalize data to [0, 1] range
            shuffle: Whether to shuffle files before processing
            max_samples: Maximum number of samples to use (for quick testing)
        """
        self.h5_files = [Path(f) for f in h5_files]
        self.batch_size = batch_size
        self.input_names = input_names or ["freq_time", "dm_time"]
        self.ft_key = ft_key
        self.dt_key = dt_key
        self.normalize = normalize
        self.shuffle = shuffle
        self.max_samples = max_samples
        
        # Statistics for normalization
        self.ft_min, self.ft_max = None, None
        self.dt_min, self.dt_max = None, None
        
        # Filter existing files
        self.valid_files = [f for f in self.h5_files if f.exists()]
        if len(self.valid_files) < len(self.h5_files):
            print(f"[WARNING] {len(self.h5_files) - len(self.valid_files)} files not found")
        
        if max_samples:
            self.valid_files = self.valid_files[:max_samples]
        
        print(f"[INIT] Loaded {len(self.valid_files)} H5 files for calibration")
        
        # Compute normalization statistics if needed
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute min/max statistics for normalization from a subset of data."""
        print("[CALIBRATION] Computing normalization statistics...")
        
        ft_vals, dt_vals = [], []
        sample_count = min(100, len(self.valid_files))  # Use up to 100 files for stats
        
        for h5_file in self.valid_files[:sample_count]:
            try:
                with h5py.File(h5_file, 'r') as f:
                    ft_data = f[self.ft_key][:]
                    dt_data = f[self.dt_key][:]
                    
                    ft_vals.extend(ft_data.flatten())
                    dt_vals.extend(dt_data.flatten())
            except Exception as e:
                print(f"Warning: Could not read {h5_file}: {e}")
                continue
        
        if ft_vals:
            self.ft_min, self.ft_max = np.min(ft_vals), np.max(ft_vals)
            self.dt_min, self.dt_max = np.min(dt_vals), np.max(dt_vals)
            print(f"[STATS] FT range: [{self.ft_min:.3f}, {self.ft_max:.3f}]")
            print(f"[STATS] DT range: [{self.dt_min:.3f}, {self.dt_max:.3f}]")
    
    def _normalize(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """Normalize data to [0, 1] range."""
        if data_max - data_min < 1e-6:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    def _load_h5_data(self, h5_file: Path) -> tuple:
        """
        Load and process data from a single H5 file.
        
        Returns:
            Tuple of (ft_data, dt_data) as numpy arrays
        """
        with h5py.File(h5_file, 'r') as f:
            # Load freq-time data
            ft_data = f[self.ft_key][:]
            dt_data = f[self.dt_key][:]
            
            # Ensure proper shape (batch, height, width, channels)
            if ft_data.ndim == 3:  # (H, W, C) or (H, W) -> add batch dim
                ft_data = np.expand_dims(ft_data, axis=0)
            if ft_data.ndim == 4 and ft_data.shape[-1] != 1:
                # Assume (B, H, W) -> add channel dim
                ft_data = np.expand_dims(ft_data, axis=-1)
            
            if dt_data.ndim == 3:
                dt_data = np.expand_dims(dt_data, axis=0)
            if dt_data.ndim == 4 and dt_data.shape[-1] != 1:
                dt_data = np.expand_dims(dt_data, axis=-1)
            
            # Ensure float32
            ft_data = ft_data.astype(np.float32)
            dt_data = dt_data.astype(np.float32)
            
            # Normalize if requested
            if self.normalize and self.ft_min is not None:
                ft_data = self._normalize(ft_data, self.ft_min, self.ft_max)
                dt_data = self._normalize(dt_data, self.dt_min, self.dt_max)
            
            return ft_data, dt_data
    
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Return iterator for calibration batches."""
        files_to_process = self.valid_files.copy()
        
        if self.shuffle:
            np.random.shuffle(files_to_process)
        
        batch_ft = []
        batch_dt = []
        batch_files = []
        
        for h5_file in files_to_process:
            try:
                ft_data, dt_data = self._load_h5_data(h5_file)
                
                # Handle different batch sizes in H5 files
                batch_size_h5 = ft_data.shape[0]
                
                for i in range(batch_size_h5):
                    batch_ft.append(ft_data[i])
                    batch_dt.append(dt_data[i])
                    batch_files.append(h5_file.name)
                    
                    if len(batch_ft) >= self.batch_size:
                        # Yield full batch
                        yield {
                            self.input_names[0]: np.stack(batch_ft, axis=0),
                            self.input_names[1]: np.stack(batch_dt, axis=0)
                        }
                        batch_ft, batch_dt, batch_files = [], [], []
                        
            except Exception as e:
                print(f"[WARNING] Failed to load {h5_file}: {e}")
                continue
        
        # Yield remaining data (last partial batch)
        if batch_ft:
            yield {
                self.input_names[0]: np.stack(batch_ft, axis=0),
                self.input_names[1]: np.stack(batch_dt, axis=0)
            }
    
    def __len__(self) -> int:
        """Return number of batches."""
        total_samples = 0
        for h5_file in self.valid_files[:100]:  # Quick estimate
            try:
                with h5py.File(h5_file, 'r') as f:
                    ft_data = f[self.ft_key][:]
                    total_samples += ft_data.shape[0]
            except:
                pass
        
        # Scale estimate
        if len(self.valid_files) > 100:
            total_samples = total_samples * len(self.valid_files) // 100
        
        return (total_samples + self.batch_size - 1) // self.batch_size


class FRBModelOptQuantizer:
    """
    ModelOpt quantizer for FRB detector with H5 calibration data support.
    Supports FP8, INT8, and INT4 quantization modes.
    """
    
    def __init__(
        self,
        onnx_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: QuantizationConfig = None
    ):
        """
        Initialize FRB quantizer.
        
        Args:
            onnx_path: Path to input ONNX model
            output_dir: Directory to save quantized models
            config: Quantization configuration
        """
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.config = config or QuantizationConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate ONNX model
        self._validate_model()
        
        # Cache directory
        self.cache_dir = self.output_dir / "calibration_cache"
        if self.config.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _validate_model(self):
        """Validate ONNX model before quantization."""
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(model)
        
        # Print model info
        print(f"\n[MODEL INFO] {self.onnx_path.name}")
        print(f"  Opset version: {model.opset_import[0].version}")
        print(f"  Inputs: {[i.name for i in model.graph.input]}")
        print(f"  Outputs: {[o.name for o in model.graph.output]}")
        
        # Store input names for calibration
        self.input_names = [i.name for i in model.graph.input]
        
    def quantize(
        self,
        calibration_h5_files: List[Union[str, Path]],
        validation_h5_files: Optional[List[Union[str, Path]]] = None
    ) -> Path:
        """
        Perform quantization on the FRB model.
        
        Args:
            calibration_h5_files: List of H5 files for calibration
            validation_h5_files: Optional H5 files for validation
            
        Returns:
            Path to quantized model
        """
        print(f"\n{'='*60}")
        print(f"Starting {self.config.mode.value.upper()} Quantization")
        print(f"{'='*60}")
        
        # Create calibration data reader
        calibration_reader = FRBCalibrationDataReader(
            h5_files=calibration_h5_files,
            batch_size=self.config.batch_size,
            input_names=self.input_names,
            ft_key="ft",  # Adjust based on your H5 file structure
            dt_key="dt",  # Adjust based on your H5 file structure
            normalize=True,
            shuffle=True
        )
        
        # Prepare quantization arguments
        quantize_args = {
            "onnx_path": str(self.onnx_path),
            "quantize_mode": self.config.mode.value,
            "calibration_data_reader": calibration_reader,
            "calibration_method": self.config.calibration_method,
            "output_path": str(self._get_output_path()),
            "op_types_to_quantize": self.config.op_types_to_quantize,
            "op_types_to_exclude": self.config.op_types_to_exclude,
        }
        
        # Add FP8-specific configurations
        if self.config.mode == QuantizationMode.FP8:
            quantize_args.update({
                "fp8_format": "e4m3",  # or "e5m2" for higher dynamic range
            })
        
        # Run quantization
        start_time = time.time()
        
        try:
            print(f"\n[QUANTIZATION] Using {len(calibration_h5_files)} H5 files")
            print(f"[QUANTIZATION] Batch size: {self.config.batch_size}")
            print(f"[QUANTIZATION] Calibration method: {self.config.calibration_method}")
            
            moq.quantize(**quantize_args)
            
            elapsed = time.time() - start_time
            print(f"\n[SUCCESS] Quantization completed in {elapsed:.2f} seconds")
            print(f"[OUTPUT] Model saved to: {self._get_output_path()}")
            
            # Validate quantized model
            quantized_path = self._validate_quantized_model()
            
            # Run validation if provided
            if validation_h5_files:
                self._validate_accuracy(validation_h5_files, quantized_path)
            
            return quantized_path
            
        except Exception as e:
            print(f"\n[ERROR] Quantization failed: {e}")
            raise
    
    def _get_output_path(self) -> Path:
        """Generate output path based on quantization mode."""
        suffix = f"_{self.config.mode.value}_{self.config.calibration_method}"
        return self.output_dir / f"frb_model{suffix}.onnx"
    
    def _validate_quantized_model(self) -> Path:
        """Validate the quantized ONNX model."""
        quantized_path = self._get_output_path()
        
        if not quantized_path.exists():
            raise RuntimeError(f"Quantized model not created: {quantized_path}")
        
        # Check if model loads
        model = onnx.load(str(quantized_path))
        onnx.checker.check_model(model)
        
        # Check file size
        size_mb = quantized_path.stat().st_size / (1024 * 1024)
        print(f"[VALIDATION] Quantized model size: {size_mb:.2f} MB")
        
        return quantized_path
    
    def _validate_accuracy(
        self,
        validation_files: List[Union[str, Path]],
        quantized_path: Path
    ):
        """
        Validate accuracy between FP32 and quantized models.
        
        Args:
            validation_files: H5 files for validation
            quantized_path: Path to quantized model
        """
        print(f"\n[VALIDATION] Running accuracy validation on {len(validation_files)} files...")
        
        # Create validation data reader
        val_reader = FRBCalibrationDataReader(
            h5_files=validation_files,
            batch_size=1,  # Use batch size 1 for validation
            input_names=self.input_names,
            normalize=True
        )
        
        # Load models
        fp32_session = ort.InferenceSession(str(self.onnx_path), providers=['CPUExecutionProvider'])
        quant_session = ort.InferenceSession(str(quantized_path), providers=['CPUExecutionProvider'])
        
        # Collect outputs
        fp32_outputs = []
        quant_outputs = []
        
        for i, batch in enumerate(val_reader):
            if i >= 50:  # Limit validation samples
                break
            
            # Run inference
            fp32_out = fp32_session.run(None, batch)[0]
            quant_out = quant_session.run(None, batch)[0]
            
            fp32_outputs.append(fp32_out)
            quant_outputs.append(quant_out)
        
        # Compute metrics
        fp32_outputs = np.concatenate(fp32_outputs, axis=0)
        quant_outputs = np.concatenate(quant_outputs, axis=0)
        
        # Calculate similarity metrics
        mse = np.mean((fp32_outputs - quant_outputs) ** 2)
        mae = np.mean(np.abs(fp32_outputs - quant_outputs))
        cos_sim = np.dot(fp32_outputs.flatten(), quant_outputs.flatten()) / (
            np.linalg.norm(fp32_outputs) * np.linalg.norm(quant_outputs)
        )
        
        print(f"\n[ACCURACY METRICS]")
        print(f"  MSE:  {mse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        
        if mse < 0.01:
            print("  ✓ Good accuracy preservation")
        elif mse < 0.1:
            print("  ⚠ Acceptable accuracy loss")
        else:
            print("  ✗ High accuracy loss - consider different calibration method")


def compare_quantization_modes(
    onnx_path: Union[str, Path],
    calibration_h5_files: List[Union[str, Path]],
    output_dir: Union[str, Path] = "quantized_models"
):
    """
    Compare different quantization modes on the FRB model.
    
    Args:
        onnx_path: Path to FP32 ONNX model
        calibration_h5_files: List of H5 files for calibration
        output_dir: Directory to save quantized models
    """
    results = {}
    
    # Test different quantization configurations
    configs = [
        # FP8 configurations
        QuantizationConfig(mode=QuantizationMode.FP8, calibration_method="entropy"),
        QuantizationConfig(mode=QuantizationMode.FP8, calibration_method="max"),
        
        # INT8 configurations
        QuantizationConfig(mode=QuantizationMode.INT8, calibration_method="entropy"),
        QuantizationConfig(mode=QuantizationMode.INT8, calibration_method="max"),
        
        # INT4 configuration (more aggressive compression)
        QuantizationConfig(mode=QuantizationMode.INT4, calibration_method="awq_clip"),
    ]
    
    for config in configs:
        print(f"\n{'#'*60}")
        print(f"Testing: {config.mode.value.upper()} with {config.calibration_method}")
        print(f"{'#'*60}")
        
        try:
            quantizer = FRBModelOptQuantizer(
                onnx_path=onnx_path,
                output_dir=Path(output_dir),
                config=config
            )
            
            quantized_path = quantizer.quantize(
                calibration_h5_files=calibration_h5_files,
                validation_h5_files=calibration_h5_files[:10]  # Use subset for validation
            )
            
            results[f"{config.mode.value}_{config.calibration_method}"] = {
                "path": str(quantized_path),
                "config": config
            }
            
        except Exception as e:
            print(f"Failed for {config.mode.value} with {config.calibration_method}: {e}")
            results[f"{config.mode.value}_{config.calibration_method}"] = {"error": str(e)}
    
    # Save comparison results
    results_path = Path(output_dir) / "quantization_comparison.json"
    with open(results_path, 'w') as f:
        # Convert non-serializable objects
        serializable_results = {}
        for k, v in results.items():
            if "path" in v:
                serializable_results[k] = {"path": v["path"]}
            else:
                serializable_results[k] = v
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")
    
    return results


# Quick test function
def quick_test():
    """
    Quick test function to verify H5 file loading and basic quantization.
    """
    # Paths - adjust these to your actual paths
    onnx_model = Path("models/frb_detector.onnx")
    h5_dir = Path("data/calibration")
    
    # Find H5 files
    h5_files = list(h5_dir.glob("*.h5"))
    
    if not onnx_model.exists():
        print(f"ERROR: ONNX model not found at {onnx_model}")
        print("Please update the path to your actual model")
        return
    
    if not h5_files:
        print(f"ERROR: No H5 files found in {h5_dir}")
        print("Please ensure H5 files exist with 'ft' and 'dt' keys")
        return
    
    print(f"Found {len(h5_files)} H5 files")
    print(f"Model: {onnx_model}")
    
    # Test FP8 quantization with smaller subset
    test_files = h5_files[:20]  # Use first 20 files for quick test
    
    config = QuantizationConfig(
        mode=QuantizationMode.FP8,
        calibration_method="entropy",
        batch_size=4,
        use_cache=True
    )
    
    quantizer = FRBModelOptQuantizer(
        onnx_path=onnx_model,
        output_dir=Path("quantized_models"),
        config=config
    )
    
    quantized_path = quantizer.quantize(
        calibration_h5_files=test_files,
        validation_h5_files=test_files[:5]
    )
    
    print(f"\n✓ Quick test completed successfully!")
    print(f"Quantized model: {quantized_path}")


if __name__ == "__main__":
    # Run quick test
    quick_test()
    
    # To compare all quantization modes, uncomment:
    # compare_quantization_modes(
    #     onnx_path="models/frb_detector.onnx",
    #     calibration_h5_files=list(Path("data/calibration").glob("*.h5")),
    #     output_dir="quantized_models"
    # )