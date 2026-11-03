# Prompt file, prompt format and decoding of every run.
# raw: the format of v1, i.e. the same Phi-3 style wrapper "<|user|>\n{prompt}<|end|>\n<|assistant|>" for all models
# chat: the chat template of each model, as stored in its GGUF file
GREEDY = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
SAMPLING = {"temperature": 0.7, "top_p": 0.9}

RUNS = {
    # v1 and the prompt format pilot (decision D1: the pilot changed the findings, so the main run uses the chat format)
    "v1": {"prompts": "output/prompts_v1.csv", "format": "raw", **GREEDY},        # original experiment (July 2026)
    "v1_1": {"prompts": "output/prompts_v1_1.csv", "format": "raw", **GREEDY},    # v1 with template 04 re-run
    "repro": {"prompts": "output/prompts_repro.csv", "format": "raw", **GREEDY},  # reproduction check of v1
    "pilot_chat": {"prompts": "output/prompts_pilot.csv", "format": "chat", **GREEDY},
    "template04_raw": {"prompts": "output/prompts_template04.csv", "format": "raw", **GREEDY},
    "template04_chat": {"prompts": "output/prompts_template04.csv", "format": "chat", **GREEDY},
    # main run
    "v2_chat": {"prompts": "output/prompts_v2.csv", "format": "chat", **GREEDY},
    # robustness runs on the pilot prompts (compared with v2_chat on the same ids)
    "name_swap_chat": {"prompts": "output/prompts_name_swap.csv", "format": "chat", **GREEDY},
    "paraphrase_chat": {"prompts": "output/prompts_paraphrase.csv", "format": "chat", **GREEDY},
    "option_order_chat": {"prompts": "output/prompts_option_order.csv", "format": "chat", **GREEDY},
    "sampling_seed1_chat": {"prompts": "output/prompts_pilot_v2.csv", "format": "chat", **SAMPLING, "seed": 1},
    "sampling_seed2_chat": {"prompts": "output/prompts_pilot_v2.csv", "format": "chat", **SAMPLING, "seed": 2},
    "sampling_seed3_chat": {"prompts": "output/prompts_pilot_v2.csv", "format": "chat", **SAMPLING, "seed": 3},
}
