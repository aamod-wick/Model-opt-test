from pathlib import Path
import os
import sys
import numpy as np
import argparse
import glob
import scipy.signal as s
import h5py
from cuda import cudart
from cuda_utilities import Common
from model_handler import download_model 
from data_handler import  h5_batch_generator
class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator using H5 files as calibration data.

    The model has two inputs:
      - freq-time (ft): shape (B, 256, 256, 1)
      - DM-time   (dt): shape (B, 256, 256, 1)

    get_batch() returns [ft_gpu_ptr, dt_gpu_ptr].
    The order of pointers matches the order TensorRT passes in `names`,
    which reflects the ONNX input order. Verify with a debug print if unsure.
    """

    def __init__(self, cache_file, h5_files, calib_batch_size=8):
        super().__init__()
        self.cache_file = cache_file
        self.calib_batch_size = calib_batch_size
        self.total = len(h5_files)
        self.processed = 0
        self.common = Common()

        # GPU allocations for both inputs — fixed shape (B, 256, 256, 1) float32
        size = int(np.dtype(np.float32).itemsize * calib_batch_size * 256 * 256 * 1)
        self.ft_allocation = self.common.cuda_call(cudart.cudaMalloc(size))
        self.dt_allocation = self.common.cuda_call(cudart.cudaMalloc(size))

        # Wire up the generator only if there are files to process
        if h5_files:
            self.batch_generator = h5_batch_generator(h5_files, batch_size=calib_batch_size)
        else:
            self.batch_generator = iter([])  # empty — will rely on cache

    def get_batch_size(self):
        return self.calib_batch_size

    def get_batch(self, names):
        """
        Called repeatedly by TensorRT until None is returned.
        `names` contains the ONNX input names in TensorRT's expected order.
        Returned pointer list must match that same order.
        """
        # Uncomment once to verify input order during first run:
        # print(f"[DEBUG] Calibration input names from TRT: {names}")

        try:
            ft_batch, dt_batch, files = next(self.batch_generator)
            self.processed += len(files)
            print(f"[CALIBRATION] Processed {self.processed} / {self.total} files")

            self.common.memcpy_host_to_device(
                self.ft_allocation, np.ascontiguousarray(ft_batch)
            )
            self.common.memcpy_host_to_device(
                self.dt_allocation, np.ascontiguousarray(dt_batch)
            )

            # Order: ft first, dt second — must match ONNX export input order
            return [int(self.ft_allocation), int(self.dt_allocation)]

        except StopIteration:
            print("[CALIBRATION] All calibration batches complete.")
            return None

    def read_calibration_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            print(f"[CALIBRATION] Loading cache from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        print(f"[CALIBRATION] Writing cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)
