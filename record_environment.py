# Record the hardware, software, model files and decoding settings used in the experiments.
# Usage: python record_environment.py      (writes results/environment.json)
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import llama_cpp

MODEL_DIR = "../../llm"
MODELS = {
    "Qwen3-4B-Instruct-2507-Q5_K_M": "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "llama-krikri-8b-instruct-q5_k_m": "https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF",
    "gemma-4-E2B-it-Q5_K_M": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def command_output(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


environment = {
    "os": platform.platform(),
    "cpu": command_output("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
    "ram_gb": round(int(command_output("grep MemTotal /proc/meminfo | awk '{print $2}'") or 0) / 1024 ** 2, 1),
    "gpu": command_output("bash -c 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; sycl-ls' | grep 'level_zero:0'"),
    "gpu_selection": "ONEAPI_DEVICE_SELECTOR=level_zero:0 (Intel Arc A770), all layers offloaded (n_gpu_layers=-1)",
    "python": platform.python_version(),
    "packages": {p: version(p) for p in ["llama_cpp_python", "pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "jinja2"]},
    "llama_cpp_backend": [f.name for f in (Path(llama_cpp.__file__).parent / "lib").glob("libggml-*.so")],
    "llama_cpp_supports_gpu_offload": bool(llama_cpp.llama_supports_gpu_offload()),
    "oneapi": command_output("ls /opt/intel/oneapi/compiler | grep -v latest"),
    "inference": {"n_ctx": 4096, "n_batch": 1024, "n_threads": 4, "n_gpu_layers": -1, "max_tokens": 30,
                  "temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.1, "seed": 42, "system_prompt": None,
                  "runs_per_prompt": 1,
                  "raw_format": "<|user|>\\n{prompt}<|end|>\\n<|assistant|> with stop <|end|> (v1, same for all models)",
                  "chat_format": "chat template stored in each GGUF file (create_chat_completion)"},
    "models": {},
}
for model_name, source in MODELS.items():
    path = Path(f"{MODEL_DIR}/{model_name}.gguf")
    print(f"Hashing {path} ...")
    environment["models"][model_name] = {"source": source, "file": path.name,
                                         "size_bytes": path.stat().st_size, "sha256": sha256(path)}

Path("results").mkdir(exist_ok=True)
with open("results/environment.json", "w", encoding="utf-8") as f:
    json.dump(environment, f, indent=2, ensure_ascii=False)
print(json.dumps(environment, indent=2, ensure_ascii=False))
