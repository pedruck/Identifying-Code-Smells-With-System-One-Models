"""Fail early when the Colab GPU packages cannot load together."""
import importlib.metadata
import sys

try:
    import torch
    from vllm.inputs import TokensPrompt  # noqa: F401 - loads vLLM CUDA extension
    from vllm import LLM  # noqa: F401 - the official encoder's runtime API
except (ImportError, OSError) as exc:
    print(
        "GPU stack import failed after setup. Restart the Colab runtime and run "
        "the latest notebook from the first cell. The pilot pins vLLM 0.11.0 "
        "and PyTorch 2.8.0 (CUDA 12.8). If it still fails, share this error "
        "and the package versions below. Original error: " + repr(exc),
        file=sys.stderr,
    )
    for name in ("torch", "vllm", "nvidia-cuda-runtime-cu12"):
        try:
            print(f"{name}={importlib.metadata.version(name)}", file=sys.stderr)
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}=missing", file=sys.stderr)
    raise SystemExit(1) from exc

print(f"GPU stack import OK: torch {torch.__version__}, CUDA {torch.version.cuda}, "
      f"vLLM {importlib.metadata.version('vllm')}")
