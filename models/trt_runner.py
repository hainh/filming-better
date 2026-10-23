"""Minimal TensorRT runner (python bindings of either `tensorrt` or `tensorrt_lean`) on top of
cudart64_12.dll via ctypes. Used by bench_engines.py (lean) and compat_check.py (full)."""
import ctypes
import time
from pathlib import Path

import numpy as np

import common
from common import MODELS


def _np_dtype(trt, dt):
    return {trt.DataType.HALF: np.float16, trt.DataType.FLOAT: np.float32}[dt]


def make_logger(trt):
    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.msgs = []

        def log(self, severity, msg):
            self.msgs.append((severity, msg))
            if severity <= trt.ILogger.WARNING:
                print(f"  [TRT {severity.name}] {msg}")
    return Logger()


class Cuda:
    def __init__(self):
        self.rt = ctypes.CDLL(str(common.ROOT / ".tools" / "cuda" / "bin" / "cudart64_12.dll"))
        self.rt.cudaGetErrorString.restype = ctypes.c_char_p

    def chk(self, rc, what=""):
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} {self.rt.cudaGetErrorString(rc).decode()} in {what}")

    def malloc(self, n):
        p = ctypes.c_void_p()
        self.chk(self.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(n, 1))), f"cudaMalloc({n})")
        return p.value

    def free(self, p):
        self.rt.cudaFree(ctypes.c_void_p(p))

    def h2d(self, dst, arr):
        arr = np.ascontiguousarray(arr)
        self.chk(self.rt.cudaMemcpy(ctypes.c_void_p(dst), arr.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_size_t(arr.nbytes), 1), "h2d")

    def d2h(self, arr, src):
        self.chk(self.rt.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src),
                                    ctypes.c_size_t(arr.nbytes), 2), "d2h")

    def stream(self):
        s = ctypes.c_void_p()
        self.chk(self.rt.cudaStreamCreate(ctypes.byref(s)))
        return s

    def sync(self, s):
        self.chk(self.rt.cudaStreamSynchronize(s), "sync")

    def mem_info(self):
        f, t = ctypes.c_size_t(), ctypes.c_size_t()
        self.chk(self.rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)))
        return f.value, t.value


class Engine:
    def __init__(self, trt, cu, path, host_code=True):
        self.trt = trt
        self.cu = cu
        self.logger = make_logger(trt)
        self.runtime = trt.Runtime(self.logger)
        self.runtime.engine_host_code_allowed = host_code
        data = Path(path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None: " + "; ".join(m for _, m in self.logger.msgs[-3:]))
        self.ctx = self.engine.create_execution_context()
        self.stream = cu.stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.OUTPUT]
        self.bufs = {}

    def describe(self):
        d = {}
        for n in self.names:
            e = self.engine
            entry = dict(mode="in" if n in self.inputs else "out", dtype=str(e.get_tensor_dtype(n)).split(".")[-1],
                         shape=list(e.get_tensor_shape(n)))
            if n in self.inputs:
                entry["profile0"] = [list(s) for s in e.get_tensor_profile_shape(n, 0)]
            d[n] = entry
        return d

    def run(self, feeds, iters=1, warmup=0):
        """feeds: name -> np array (engine dtype). Returns (outputs dict, best-of-iters ms); see self.last_stats."""
        cu, ctx = self.cu, self.ctx
        for n in self.inputs:
            assert ctx.set_input_shape(n, list(feeds[n].shape)), f"set_input_shape {n} {feeds[n].shape}"
        need = {}
        for n in self.inputs + self.outputs:
            shp = tuple(feeds[n].shape) if n in self.inputs else tuple(ctx.get_tensor_shape(n))
            assert all(s > 0 for s in shp), (n, shp)
            need[n] = (shp, _np_dtype(self.trt, self.engine.get_tensor_dtype(n)))
        for n, (shp, dt) in need.items():
            nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
            if n not in self.bufs or self.bufs[n][1] < nbytes:
                if n in self.bufs:
                    cu.free(self.bufs[n][0])
                self.bufs[n] = (cu.malloc(nbytes), nbytes)
            ctx.set_tensor_address(n, self.bufs[n][0])
        for n in self.inputs:
            cu.h2d(self.bufs[n][0], feeds[n].astype(need[n][1]))
        for _ in range(warmup):
            assert ctx.execute_async_v3(self.stream.value)
        cu.sync(self.stream)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            assert ctx.execute_async_v3(self.stream.value)
            cu.sync(self.stream)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        self.last_stats = dict(min=times[0], median=times[len(times) // 2], mean=sum(times) / len(times))
        ms = self.last_stats["min"]  # best-of-N: the GPU may be shared, min approximates an idle GPU
        outs = {}
        for n in self.outputs:
            shp, dt = need[n]
            a = np.empty(shp, dtype=dt)
            cu.d2h(a, self.bufs[n][0])
            outs[n] = a
        return outs, ms

    def close(self):
        for p, _ in self.bufs.values():
            self.cu.free(p)
        self.bufs = {}
        del self.ctx, self.engine, self.runtime


def random_feeds(mid, h, w, dt=np.float16):
    rng = np.random.default_rng(0)
    if MODELS[mid]["kind"] == "upscale":
        return {"input": rng.random((1, 3, h, w), dtype=np.float32).astype(dt)}
    return {"img0": rng.random((1, 3, h, w), dtype=np.float32).astype(dt),
            "img1": rng.random((1, 3, h, w), dtype=np.float32).astype(dt),
            "timestep": np.full((1, 1, 1, 1), 0.5, dtype=dt)}
