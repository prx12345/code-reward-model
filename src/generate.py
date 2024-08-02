"""Sample N candidates per problem. Resumable: tops up whatever is on disk."""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any

import config
from src.utils import JsonlAppender, log, progress, read_jsonl

_FENCED = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_UNCLOSED_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*)", re.DOTALL | re.IGNORECASE)
_CODE_START = re.compile(r"^(?:from |import |def |class |@)", re.MULTILINE)


# --------------------------------------------------------------------------
# Response -> code
# --------------------------------------------------------------------------

def extract_code(response: str) -> str:
    """Pull runnable Python out of a chat model's reply.

    Order of attempts, each a real failure mode observed with small instruct
    models: (1) a proper fenced block; (2) a fence the model opened and never
    closed because it hit ``max_new_tokens``; (3) no fence at all, in which
    case we take everything from the first import/def/decorator onward and
    drop trailing prose. Returning the raw reply on total failure is
    deliberate: it will be labeled ``syntax_error`` by the executor, which is
    the honest label for "the model did not produce code".
    """
    blocks = _FENCED.findall(response)
    if blocks:
        # Last block: models often show a wrong sketch first, then the answer.
        return blocks[-1].strip("\n")
    unclosed = _UNCLOSED_FENCE.search(response)
    if unclosed:
        return unclosed.group(1).strip("\n")
    match = _CODE_START.search(response)
    if match:
        tail = response[match.start():]
        lines = tail.split("\n")
        # Trim trailing natural-language lines (no indent, no code punctuation).
        while lines and lines[-1].strip() and not lines[-1].startswith((" ", "\t")) \
                and not _CODE_START.match(lines[-1]) and lines[-1].strip()[-1] not in "):]}\"'":
            lines.pop()
        return "\n".join(lines).strip("\n")
    return response.strip()


def _meta_record(model_name: str, generator_tag: str, benchmark: str,
                 n_problems: int) -> dict[str, Any]:
    return {
        "record_type": "meta",
        "model": model_name,
        "generator_tag": generator_tag,
        "benchmark": benchmark,
        "n_problems": n_problems,
        "n_candidates_per_problem": config.N_CANDIDATES,
        "temperature": config.GEN_TEMPERATURE,
        "top_p": config.GEN_TOP_P,
        "max_new_tokens": config.GEN_MAX_NEW_TOKENS,
        "dtype": config.GEN_DTYPE,
        "seed": config.SEED,
    }


def load_generations(path: str | Path) -> list[dict[str, Any]]:
    """Candidates only -- the meta line is filtered out."""
    return [r for r in read_jsonl(path) if r.get("record_type") != "meta"]


def generation_meta(path: str | Path) -> dict[str, Any] | None:
    for record in read_jsonl(path):
        if record.get("record_type") == "meta":
            return record
    return None


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_candidates(problems: list[dict[str, Any]], model_name: str,
                        generator_tag: str, benchmark: str,
                        n_candidates: int | None = None,
                        limit: int = 0) -> Path:
    """Sample candidates for every problem, resuming from whatever is on disk."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils import set_seed

    n_candidates = n_candidates or config.N_CANDIDATES
    out_path = config.generations_path(generator_tag, benchmark)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if limit:
        problems = problems[:limit]

    existing = load_generations(out_path)
    have = collections.Counter(r["problem_id"] for r in existing)
    max_index = collections.defaultdict(int)
    for record in existing:
        max_index[record["problem_id"]] = max(
            max_index[record["problem_id"]], record["candidate_index"] + 1
        )
    todo = [p for p in problems if have[p["problem_id"]] < n_candidates]
    log(f"generation: {len(problems) - len(todo)}/{len(problems)} problems complete, "
        f"{len(todo)} to do -> {out_path.name}")

    writer = JsonlAppender(out_path)
    if generation_meta(out_path) is None:
        writer.write(_meta_record(model_name, generator_tag, benchmark, len(problems)))
    if not todo:
        writer.close()
        return out_path

    set_seed(config.SEED)
    log(f"loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must be LEFT-padded for batched generation; with
    # right padding the model attends to pad tokens before the real prompt and
    # the samples quietly degrade. This is the single easiest way to get
    # garbage candidates and then blame the model for them.
    tokenizer.padding_side = "left"

    dtype = getattr(torch, config.GEN_DTYPE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    try:
        for problem in progress(todo, total=len(todo), every=10, label="gen"):
            needed = n_candidates - have[problem["problem_id"]]
            start_index = max_index[problem["problem_id"]]
            messages = [{"role": "user", "content": problem["generation_prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(device)

            produced = 0
            while produced < needed:
                batch = min(config.GEN_BATCH_SIZE, needed - produced)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=config.GEN_TEMPERATURE,
                        top_p=config.GEN_TOP_P,
                        max_new_tokens=config.GEN_MAX_NEW_TOKENS,
                        num_return_sequences=batch,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                for offset in range(batch):
                    completion = tokenizer.decode(
                        outputs[offset][prompt_len:], skip_special_tokens=True
                    )
                    index = start_index + produced + offset
                    writer.write({
                        "candidate_id": f"{problem['problem_id']}#{generator_tag}#{index}",
                        "problem_id": problem["problem_id"],
                        "benchmark": problem["benchmark"],
                        "generator": generator_tag,
                        "candidate_index": index,
                        "code": extract_code(completion),
                        "raw_response": completion[:8000],
                    })
                produced += batch
    finally:
        writer.close()
        log(f"generation checkpointed to {out_path}")

    return out_path
