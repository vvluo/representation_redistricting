"""ctypes bridge to the C++ ZDD builder (cpp/zdd_build.cpp).

The shared library is compiled on first use (or when the source is newer) with the
system C++ compiler:  c++ -O3 -std=c++17 -shared -fPIC.  Set $CXX to override.
    python -m redistricting.enumerate._cpp      # compile now and report
"""
import ctypes, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent / "cpp"
SRC = HERE / "zdd_build.cpp"
LIB = HERE / ("_zdd_build.dylib" if sys.platform == "darwin" else "_zdd_build.so")
STATUS = {1: "node budget", 2: "time budget", -2: "a vertex has no edge",
          -3: "frontier wider than 62 vertices"}
_lib = None


def compile_lib(verbose=False):
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        raise RuntimeError("no C++ compiler found; set $CXX or use backend='python'")
    cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-o", str(LIB), str(SRC)]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def lib():
    global _lib
    if _lib is None:
        if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
            compile_lib()
        L = ctypes.CDLL(str(LIB))
        L.zdd_build.restype = ctypes.c_void_p
        L.zdd_build.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                ctypes.c_longlong, ctypes.c_double]
        for f in ("zdd_status", "zdd_levels"):
            getattr(L, f).restype = ctypes.c_int; getattr(L, f).argtypes = [ctypes.c_void_p]
        L.zdd_nodes.restype = ctypes.c_longlong; L.zdd_nodes.argtypes = [ctypes.c_void_p]
        L.zdd_width.restype = ctypes.c_longlong; L.zdd_width.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.zdd_copy.restype = None
        L.zdd_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_int32)]
        L.zdd_free.restype = None; L.zdd_free.argtypes = [ctypes.c_void_p]
        _lib = L
    return _lib


def build(n, order, p, max_nodes=None, max_seconds=None):
    """Return (lo list, hi list, widths, nodes, status, levels_built)."""
    L = lib()
    eu = np.ascontiguousarray([u for u, v in order], dtype=np.int32)
    ev = np.ascontiguousarray([v for u, v in order], dtype=np.int32)
    P = ctypes.POINTER(ctypes.c_int)
    h = L.zdd_build(n, len(order), eu.ctypes.data_as(P), ev.ctypes.data_as(P), p,
                    int(max_nodes or 0), float(max_seconds or 0))
    try:
        st = L.zdd_status(h); nl = L.zdd_levels(h)
        lo, hi, width = [], [], []
        P32 = ctypes.POINTER(ctypes.c_int32)
        for l in range(nl):
            w = L.zdd_width(h, l)
            a = np.empty(w, dtype=np.int32); b = np.empty(w, dtype=np.int32)
            L.zdd_copy(h, l, a.ctypes.data_as(P32), b.ctypes.data_as(P32))
            lo.append(a); hi.append(b); width.append(int(w))
        return lo, hi, width, int(L.zdd_nodes(h)), st, nl
    finally:
        L.zdd_free(h)


if __name__ == "__main__":
    compile_lib(verbose=True)
    print("built", LIB)
