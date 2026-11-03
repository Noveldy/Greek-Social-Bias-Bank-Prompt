# Qwen retrieved from https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/tree/main
# Krikri retrieved from https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct-GGUF/tree/main
# Gemma retrieved from https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main
# Usage: python run_inference.py <model_name> <run_name>
#   e.g. python run_inference.py Qwen3-4B-Instruct-2507-Q5_K_M pilot_chat
import os
# run only on the Intel Arc A770 (SYCL device level_zero:0); level_zero:1 is the integrated GPU
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
import csv
import sys
import time
from pathlib import Path
import pandas as pd
import llama_cpp
from llama_cpp import Llama
from config.runs import RUNS

MODEL_DIR = "../../llm"  # folder with the GGUF files
MODEL_NAME = sys.argv[1]
RUN_NAME = sys.argv[2]

# prompt file, prompt format and decoding of the run (see config/runs.py)
PROMPTS_FILE = RUNS[RUN_NAME]["prompts"]
PROMPT_FORMAT = RUNS[RUN_NAME]["format"]
TEMPERATURE = RUNS[RUN_NAME]["temperature"]
TOP_P = RUNS[RUN_NAME]["top_p"]
SEED = RUNS[RUN_NAME]["seed"]  # seed of the run, see prompt_seed()
OUTPUT_FILE = f"output/llm_responses_{MODEL_NAME}_{RUN_NAME}.csv"

# decoding parameters common to all runs (same as v1)
MAX_TOKENS = 30
TOP_K = 40  # llama-cpp-python default, only used when sampling
REPEAT_PENALTY = 1.1

if not llama_cpp.llama_supports_gpu_offload():
    sys.exit("llama-cpp-python was built without GPU support, see README")


def remove_incomplete_last_line(path):
    # a power cut while writing can leave the last line incomplete: remove it so that its prompt is asked again
    data = Path(path).read_bytes()
    complete = data.rstrip(b"\x00")
    if not complete.endswith(b"\n"):
        complete = complete[: complete.rfind(b"\n") + 1]
    if complete != data:
        print(f"Removed an incomplete last line from {path}")
        if complete:
            Path(path).write_bytes(complete)
        else:
            Path(path).unlink()


# load the prompts and skip the ones that already have a response (to resume an interrupted run)
prompts = pd.read_csv(Path(PROMPTS_FILE)).to_dict(orient="records")
done_ids = set()
if Path(OUTPUT_FILE).exists():
    remove_incomplete_last_line(OUTPUT_FILE)
if Path(OUTPUT_FILE).exists():
    done_ids = set(pd.read_csv(Path(OUTPUT_FILE))["id"])
prompts = [p for p in prompts if p["id"] not in done_ids]
print(f"Run {RUN_NAME} ({PROMPT_FORMAT} format), model {MODEL_NAME}: {len(prompts)} prompts left")
if not prompts:
    sys.exit(0)  # run already complete, no need to load the model

# load the local LLM (all layers on the GPU)
llm = Llama(
    model_path=f"{MODEL_DIR}/{MODEL_NAME}.gguf",
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=-1,
    n_batch=1024,
    split_mode=0,  # don't split across GPUs
    main_gpu=0,
    seed=SEED,
    verbose=True,  # prints the GPU device and the number of offloaded layers
)


def prompt_seed(id):
    # llama-cpp-python restarts the random generator from the given seed at every call, so the same seed for all prompts
    # would give every prompt the same random draws. Each prompt gets its own seed instead, derived from the seed of the
    # run and the prompt id, so answers do not depend on the order of the prompts (a resumed run gives the same answers).
    # Greedy runs (temperature 0) do not use the seed.
    return SEED * 10_000_000 + id


# function to query a local LLM
def query_LLM(id, prompt):
    start_time = time.time()
    if PROMPT_FORMAT == "raw":
        output = llm(
            f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
            seed=prompt_seed(id),
            stop=["<|end|>"],
            echo=False,
        )
        final_answer = output["choices"][0]["text"]
    else:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
            seed=prompt_seed(id),
        )
        final_answer = output["choices"][0]["message"]["content"]
    inference_time = time.time() - start_time

    response = {
        "id": id,
        "prompt": prompt,
        "model_name": MODEL_NAME,
        "inference_time": inference_time,
        "finish_reason": output["choices"][0]["finish_reason"],
        "final_answer": (final_answer or "").replace("\n", " "),  # remove new lines from response
    }
    return response


for i, p in enumerate(prompts):
    response = query_LLM(p["id"], p["prompt"])
    write_header = not Path(OUTPUT_FILE).exists()
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, response.keys())
        if write_header:
            w.writeheader()
        w.writerow(response)
        f.flush()
        os.fsync(f.fileno())  # write the answer to the disk now, so that a power cut loses at most the current prompt
    if i % 100 == 0:
        print(f"| Prompt {i+1}/{len(prompts)} | ID: {response['id']} | Inference time: {response['inference_time']:.3f} s | Finish reason: {response['finish_reason']} |", flush=True)
print("Done")
