#!/usr/bin/env python3
"""FP8 Quantization for CNN with H5 calibration data"""

import argparse
import numpy as np
import h5py
import modelopt.onnx.quantization as moq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="fp8", choices=["fp8", "int8"])
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    
    # Create batcher
    class H5Batcher:
        def __init__(self, h5_path, num_samples):
            self.h5 = h5py.File(h5_path, 'r')
            self.data = self.h5['data']  # Adjust key
            self.num_samples = min(len(self.data), num_samples)
            self.idx = 0
        
        def __iter__(self):
            self.idx = 0
            return self
        
        def __next__(self):
            if self.idx >= self.num_samples:
                raise StopIteration
            sample = self.data[self.idx]
            self.idx += 1
            return {"input": np.expand_dims(sample, axis=0).astype(np.float32)}
    
    # Quantize
    batcher = H5Batcher(args.h5, args.samples)
    moq.quantize(
        onnx_path=args.onnx,
        quantize_mode=args.mode,
        calibration_data=batcher,
        output_path=args.output
    )
    print(f"Done! Quantized model saved to {args.output}")

if __name__ == "__main__":
    main()