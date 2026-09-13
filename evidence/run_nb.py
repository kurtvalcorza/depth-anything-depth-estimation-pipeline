import argparse, faulthandler, io, json, os, re, sys, time, traceback, types
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

faulthandler.enable()

# Prevent stale pre-installed triton from crashing with CUDA ABI mismatch
sys.modules["triton"] = None
sys.modules["triton._C"] = None
sys.modules["triton._C.libtriton"] = None

class TeeIO(io.StringIO):
    def __init__(self, target):
        super().__init__()
        self._target = target
    def write(self, s):
        super().write(s)
        try:
            self._target.write(s)
            self._target.flush()
        except Exception:
            pass
    def flush(self):
        super().flush()
        try:
            self._target.flush()
        except Exception:
            pass

def install_colab_shim(upload_path: str | None, log: list):
    try:
        import google
    except ImportError:
        google = types.ModuleType("google")
        google.__path__ = []
        sys.modules["google"] = google

    class FlexibleModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            mod = FlexibleModule(f"{self.__name__}.{name}")
            mod.__path__ = []
            setattr(self, name, mod)
            sys.modules[f"{self.__name__}.{name}"] = mod
            return mod
        def __call__(self, *args, **kwargs):
            return None

    colab = FlexibleModule("google.colab")
    colab.__path__ = []
    files = types.ModuleType("google.colab.files")
    userdata = types.ModuleType("google.colab.userdata")
    output = FlexibleModule("google.colab.output")
    output.__path__ = []

    def upload():
        if not upload_path:
            raise RuntimeError("shim: files.upload() called but no --upload given")
        p = Path(upload_path)
        log.append({"shim": "files.upload", "name": p.name, "bytes": p.stat().st_size})
        return {p.name: p.read_bytes()}

    def download(path):
        log.append({"shim": "files.download", "path": str(path), "bytes": Path(path).stat().st_size})

    class SecretNotFoundError(Exception):
        pass

    def get(name):
        v = os.environ.get(name)
        if v is None:
            raise SecretNotFoundError(name)
        log.append({"shim": "userdata.get", "name": name})
        return v

    files.upload, files.download = upload, download
    userdata.get, userdata.SecretNotFoundError = get, SecretNotFoundError
    colab.files, colab.userdata, colab.output = files, userdata, output
    google.colab = colab

    import importlib.machinery as _m
    for _mod in (colab, files, userdata, output):
        _mod.__spec__ = _m.ModuleSpec(_mod.__name__, None, is_package=True)

    sys.modules.update({
        "google.colab": colab,
        "google.colab.files": files,
        "google.colab.userdata": userdata,
        "google.colab.output": output
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--report", required=True)
    ap.add_argument("--upload")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()

    nb = json.loads(Path(a.notebook).read_text(encoding="utf-8"))
    shim_log = []
    install_colab_shim(a.upload, shim_log)
    overrides = {}
    for s in a.set:
        k, v = s.split("=", 1)
        overrides[k] = v

    import pandas as pd
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 250)

    def display(x):
        if isinstance(x, pd.DataFrame):
            print(x.to_string())
        else:
            print(repr(x))

    ns = {"__name__": "__main__", "display": display}
    params = dict(x.split("=", 1) for x in a.param)
    report = {"notebook": a.notebook, "cells": [], "shim": shim_log, "overrides": overrides, "params": params, "params_applied": [], "ok": True}
    code_ord = -1
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_ord += 1
        src = "".join(cell["source"])
        rec = {"cell_index": idx, "code_ordinal": code_ord}
        lines = src.splitlines()
        for pi, line in enumerate(lines):
            for k, v in params.items():
                if re.match(rf"^{re.escape(k)}\s*=.*#\s*@param", line):
                    lines[pi] = f"{k} = {v} " + line[line.index("# @param"):]
                    report["params_applied"].append({"cell": idx, "line": lines[pi]})
        src = "\n".join(lines)
        if any(l.lstrip().startswith(("%", "!")) for l in lines):
            magics = [l for l in lines if l.lstrip().startswith(("%", "!"))]
            rec.update({"skipped_magics": magics})
            src = "\n".join(l for l in lines if not l.lstrip().startswith(("%", "!")))
        out = TeeIO(sys.__stdout__)
        err = TeeIO(sys.__stderr__)
        print(f"=== cell {idx} (code #{code_ord}) START ===", flush=True)
        t0 = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(compile(src, f"<cell {idx}>", "exec"), ns)
                assigned = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", src, re.M))
                for k, v in overrides.items():
                    if k in assigned:
                        ns[k] = eval(v, ns)
                        print(f"[override] {k} = {v!r}")
            rec["status"] = "ok"
        except BaseException as e:
            rec["status"] = "error"
            rec["exception"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            report["ok"] = False
        rec["wall_s"] = round(time.time() - t0, 2)
        rec["stdout"] = out.getvalue()
        rec["stderr"] = err.getvalue()[-6000:]
        try:
            import torch
            if torch.cuda.is_available():
                rec["cuda_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                rec["cuda_alloc_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        except Exception:
            pass
        report["cells"].append(rec)
        print(f"=== cell {idx} (code #{code_ord}) {rec['status']} {rec['wall_s']}s ===", flush=True)
        if rec["status"] == "error":
            print(rec["traceback"], flush=True)
            break
        if a.stop_after is not None and code_ord >= a.stop_after:
            break
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("REPORT", a.report, "OK" if report["ok"] else "FAILED")
    sys.exit(0 if report["ok"] else 1)

if __name__ == "__main__":
    main()
