# -*- coding: utf-8 -*-
import argparse
import io
import os
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterable, Optional

import numpy as np
from scipy import linalg
from tqdm.auto import tqdm

# ==============================================================================
# ==============================================================================
import torch
import torch.nn.functional as F
from pytorch_fid.inception import InceptionV3

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    IS_NPU_AVAILABLE = True
except ImportError:
    IS_NPU_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference batch npz file")
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()

    evaluator = Evaluator()

    print("computing reference batch activations...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("computing/reading reference batch statistics...")
    ref_stats = evaluator.read_statistics(args.ref_batch, ref_acts)[0]

    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("computing/reading sample batch statistics...")
    sample_stats = evaluator.read_statistics(args.sample_batch, sample_acts)[0]

    print("\nComputing FID...")
    print("FID:", sample_stats.frechet_distance(ref_stats))


class FIDStatistics:
    """Frechet distance implementation based on the pytorch-fid formulation."""
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(self, session=None, batch_size=64):
        """
        PyTorch evaluation backend. The session argument is retained for API compatibility.
        """
        self.batch_size = batch_size
        
        self.device = torch.device("npu:0" if IS_NPU_AVAILABLE else "cuda:0" if torch.cuda.is_available() else "cpu")
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"PyTorch evaluation engine active on: {self.device}")

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid_model = InceptionV3([block_idx], normalize_input=True).to(self.device)
        self.fid_model.eval()

    def warmup(self):
        pass

    def read_activations(self, npz_path: str):
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]):
        acts = []
        for batch in tqdm(batches, desc="Extracting Features (PyTorch)", leave=False):
            batch_tensor = torch.tensor(batch, device=self.device).permute(0, 3, 1, 2).float() / 255.0
            
            with torch.no_grad():
                pred = self.fid_model(batch_tensor)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
                feat_2048 = pred.squeeze(3).squeeze(2)
                
            acts.append(feat_2048.cpu().numpy())
            
        return (np.concatenate(acts, axis=0),)

    def read_statistics(self, npz_path: str, activations):
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            stat_pool = FIDStatistics(obj["mu"], obj["sigma"])
            return (stat_pool,)
            
        stat_pool = self.compute_statistics(activations[0])
        return (stat_pool,)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

# ==============================================================================
# ==============================================================================
class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]: pass

    @abstractmethod
    def remaining(self) -> int: pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None: break
                yield batch
        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)

class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length
    def __len__(self): return self.length
    def __iter__(self): return self.gen_fn()

class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]: return None
        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs
        if self.dtype.itemsize == 0: return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)
        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int: return max(0, self.shape[0] - self.idx)

class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f: arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]: return None
        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int: return max(0, self.arr.shape[0] - self.idx)

@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0): header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0): header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject: yield MemoryNpzArrayReader.load(path, arr_name)
        else: yield StreamingNpzArrayReader(arr_f, shape, dtype)

def _read_bytes(fp, size, error_template="ran out of data"):
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size: break
        except io.BlockingIOError: pass
    if len(data) != size: raise ValueError("EOF")
    return data

@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist(): raise ValueError(f"missing {arr_name} in npz file")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f: yield arr_f


if __name__ == "__main__":
    main()
