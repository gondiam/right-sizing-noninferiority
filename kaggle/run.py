"""Score MedQA with every model configuration. Kaggle script kernel, one T4, no repo clone.

push.sh fills in the placeholders. For each item the model sees the question and the four
options, and the reply is forced to start with "Answer:". The prediction is the letter with
the highest next-token probability among A, B, C and D (greedy decoding restricted to the
four options), so every item gets an answer. The probability of that letter is the
confidence used by the cascade.
"""

import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path

SPLIT, LIMIT = "@SPLIT@", int("@LIMIT@")  # dev | test, 0 = all items
COMMIT = "@COMMIT@"
DATASET = ("GBaker/MedQA-USMLE-4-options", "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18")
MODELS = {
    "1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    "8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
}
CONFIGS = ["1.7b-4bit", "4b-4bit", "4b-fp16", "8b-4bit"]  # the 8B in fp16 does not fit a T4
N_DEV, LETTERS = 500, "ABCD"
PROMPT = (
    "The following is a multiple-choice question from a medical licensing exam. "
    "Answer with the letter of the correct option.\n\n{question}\n\n{options}"
)
OUT = Path("/kaggle/working/results") / SPLIT

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "bitsandbytes"],
    check=True,
)

import bitsandbytes  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402


def items() -> list[dict]:
    """dev: 500 train items drawn with seed 0 (the dataset has no dev split). test: all 1,273."""
    if SPLIT == "test":
        ds = load_dataset(DATASET[0], revision=DATASET[1], split="test")
        idx = range(len(ds))
    else:
        ds = load_dataset(DATASET[0], revision=DATASET[1], split="train")
        idx = sorted(random.Random(0).sample(range(len(ds)), N_DEV))
    rows = [{"id": f"{SPLIT}-{i}", **ds[i]} for i in idx]
    return rows[:LIMIT] if LIMIT else rows


def score(config: str, rows: list[dict]) -> dict:
    size, quant = config.split("-")
    name, rev = MODELS[size]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    tok.padding_side = "left"
    kw = (
        {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        }
        if quant == "4bit"
        else {"dtype": torch.float16}
    )
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, device_map={"": 0}, **kw)
    model.eval()
    letter_ids = [tok.encode(" " + x, add_special_tokens=False) for x in LETTERS]
    assert all(len(t) == 1 for t in letter_ids), letter_ids  # " A" must be one token
    letter_ids = [t[0] for t in letter_ids]

    def prompt(r: dict) -> str:
        options = "\n".join(f"{x}. {r['options'][x]}" for x in LETTERS)
        messages = [
            {"role": "user", "content": PROMPT.format(question=r["question"], options=options)}
        ]
        return (
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            + "Answer:"
        )

    prompts = [prompt(r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))  # less padding per batch
    probs: list = [None] * len(rows)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for s in range(0, len(order), 8):
        idx = order[s : s + 8]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, letter_ids].float()
        for i, p in zip(idx, torch.softmax(logits, -1).tolist(), strict=True):
            probs[i] = p
    seconds = time.time() - start
    with (OUT / f"{config}.jsonl").open("w") as f:
        for r, p in zip(rows, probs, strict=True):
            pred = LETTERS[max(range(4), key=p.__getitem__)]
            rec = {
                "id": r["id"],
                "gold": r["answer_idx"],
                "pred": pred,
                "probs": [round(x, 4) for x in p],
            }
            f.write(json.dumps(rec) + "\n")
    acc = sum(
        LETTERS[max(range(4), key=p.__getitem__)] == r["answer_idx"]
        for r, p in zip(rows, probs, strict=True)
    )
    stats = {
        "model": name,
        "revision": rev,
        "items": len(rows),
        "accuracy": acc / len(rows),
        "seconds": round(seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "example_prompt": prompts[0],
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return stats


OUT.mkdir(parents=True, exist_ok=True)
rows = items()
log: dict = {
    "commit": COMMIT,
    "split": SPLIT,
    "gpu": torch.cuda.get_device_name(0),
    "versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    },
    "configs": {},
}
for config in CONFIGS:
    try:
        log["configs"][config] = score(config, rows)
    except Exception as e:  # keep the other configurations if one does not fit
        log["configs"][config] = {"error": repr(e)}
    print(
        config,
        {k: v for k, v in log["configs"][config].items() if k != "example_prompt"},
        flush=True,
    )
    (OUT / "run.json").write_text(json.dumps(log, indent=1))
