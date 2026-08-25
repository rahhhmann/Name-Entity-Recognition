"""
utils_ner.py
============
Orchestration layer and validation testbed for the Bangla Dialect
Named Entity Recognition (NER) Pipeline ONLY.
(MT/translation logic lives separately in mt_pipeline/utils.py)

Note: the NER prep script (01_prepare_ner.py) reads CSVs directly with
pandas and does not need the Excel-cleaning helpers used by the MT
pipeline, so this module only keeps what NER actually uses:
JSONL validation + a single run_script_via_exec/run_all orchestrator.
"""
import json
import time
import shutil
from pathlib import Path


def validate_jsonl(path: str | Path, required_keys: list[str] | None = None) -> tuple[int, int]:
    required_keys = required_keys or []
    path = Path(path)
    n_valid, n_invalid = 0, 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                missing = [k for k in required_keys if k not in obj]
                if missing:
                    n_invalid += 1
                else:
                    n_valid += 1
            except json.JSONDecodeError:
                n_invalid += 1
    return n_valid, n_invalid


def run_script_via_exec(script_name: str, scripts_dir: Path) -> None:
    filepath = scripts_dir / script_name
    if not filepath.exists():
        raise FileNotFoundError(f"[utils_ner] Script not found: {filepath}")

    print(f"[utils_ner] Executing: {script_name}")
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    global_context = {"__file__": str(filepath)}
    exec(code, global_context)


def run_all():
    """Runs the FULL NER pipeline only: 01_prepare_ner.py."""
    scripts = Path("scripts")
    proc = Path("data_processed")

    if proc.exists():
        print("[utils_ner] Purging old data_processed/ folder to avoid stale cache...")
        shutil.rmtree(proc)
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "ner").mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}\n# STARTING NER DATA PIPELINE\n{'#'*60}\n")
    t_start = time.time()

    pipeline_flow = [
        "01_prepare_ner.py",
    ]

    for script_file in pipeline_flow:
        run_script_via_exec(script_file, scripts)

    elapsed = time.time() - t_start
    print(f"\n{'#'*60}\n# NER PIPELINE EXECUTED SUCCESSFULLY ({elapsed:.1f}s)\n{'#'*60}")

    print("\n[utils_ner] Validating output JSONL files ...")
    files_to_check = [
        (proc / "ner" / "ner_combined.jsonl", ["dialect", "tokens", "tags"]),
        (proc / "ner" / "train_ner.jsonl", ["dialect", "tokens", "tags"]),
        (proc / "ner" / "val_ner.jsonl", ["dialect", "tokens", "tags"]),
        (proc / "ner" / "test_ner.jsonl", ["dialect", "tokens", "tags"]),
    ]

    for fpath, keys in files_to_check:
        if fpath.exists():
            v, inv = validate_jsonl(fpath, keys)
            print(f" -> {fpath.relative_to(proc.parent)}: Valid={v}, Invalid={inv}")
        else:
            print(f" -> WARNING: Missing expected file {fpath.relative_to(proc.parent)}")


if __name__ == "__main__":
    run_all()
