#!/usr/bin/env python3
"""Remote one-shot payload for colab run; stream every completed record home."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def build_command(bundle, python, script, prompts, output, items=None, pairs=None):
    """The exact remote invocation, driven by the bundle (frozen in bundle.json at prepare time)
    so arms, caps and reader travel with the run instead of living here. Two jobs: 'generate'
    (generate.py over bundle['generation']) and 'typicality' (typicality.py over bundle['items']
    under one reference model, completion-prefix frame, per-token losses, refuse truncation)."""
    job = bundle.get("job", {"kind": "generate"})
    if job["kind"] == "judge":
        return [python, str(script), "--model", bundle["model"], "--judge-lineage", job["judge_lineage"],
                "--judge-stage", job["judge_stage"], "--items", str(items), "--prompts", str(prompts),
                "--pairs-file", str(pairs), "--criterion", job["criterion"], "--out", str(output),
                "--max-len", str(job["max_len"])]
    if job["kind"] == "typicality":
        return [python, str(script), "--model", bundle["model"], "--model-tag", job["ref_tag"],
                "--items", str(items), "--out", str(output),
                "--frame", "completion-prefix", "--prompts", str(prompts),
                "--max-len", str(job["max_len"]), "--save-token-nll", "--fail-on-truncation"]
    gen = bundle["generation"]
    return [python, str(script), "--model", bundle["model"],
            "--lineage", "zephyr", "--stage", "dpo", "--prompts", str(prompts),
            "--out", str(output), "--k", str(gen["k"]),
            "--max-new-tokens", str(gen["max_new_tokens"]),
            "--elicitation", gen["elicitation"], "--k-start", str(gen.get("k_start", 0)),
            "--temperature", str(gen.get("temperature", 1.0)), "--top-p", str(gen.get("top_p", 0.95)),
            "--seed", str(gen.get("seed", 411)), "--run-id", gen.get("run_id") or bundle["run_id"], "--device", "cuda"]


def main():
    bundle = json.loads(base64.b64decode(sys.argv[1]))
    source = bundle["source"]
    if hashlib.sha256(source.encode()).hexdigest() != bundle["source_sha256"]:
        raise ValueError("generation source checksum mismatch")
    job = bundle.get("job", {"kind": "generate"})
    expected = job.get("expected_records") or (sum(g["expected_records"] for g in bundle["generations"])
                                                if bundle.get("generations") else bundle["generation"]["expected_records"])
    work = Path("/content") / bundle["run_id"]
    work.mkdir(exist_ok=False)
    script, prompts, output = work / bundle.get("script_name", "generate.py"), work / "prompts.jsonl", work / "outputs.jsonl"
    script.write_text(source)
    prompts.write_text("".join(json.dumps(p) + "\n" for p in bundle["prompts"]))
    items = None
    if "items" in bundle:
        items = work / "items.jsonl"
        items.write_text("".join(json.dumps(i) + "\n" for i in bundle["items"]))
    pairs = None
    if "pairs" in bundle:
        pairs = work / "pairs.jsonl"
        pairs.write_text("".join(json.dumps(p) + "\n" for p in bundle["pairs"]))
    import importlib.metadata
    print("B0_ENV " + json.dumps({p: importlib.metadata.version(p) for p in ("torch", "transformers")}), flush=True)
    # A sweep session runs several generation specs back to back (one dose per spec, each with its
    # own run_id and output file) so the weight download is paid once. Records stream as they land.
    if job["kind"] == "generate" and bundle.get("generations"):
        runs = [({**bundle, "generation": g}, work / f"outputs-{i}.jsonl") for i, g in enumerate(bundle["generations"])]
    else:
        runs = [(bundle, output)]
    printed_total = 0
    for i, (b, out_path) in enumerate(runs):
        command = build_command(b, sys.executable, script, prompts, out_path, items, pairs)
        printed_total += stream_run(command, out_path, work / f"generation-{i}.log")
    if printed_total != expected:
        raise RuntimeError(f"expected {expected} records, saw {printed_total}")
    print(f"B0_DONE {expected}", flush=True)


def stream_run(command, output, logpath):
    """Run one command, streaming every completed output line home; returns the count."""
    printed, last_status = 0, 0
    with logpath.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while True:
            rc = proc.poll()  # if exited, all file writes precede the final drain below
            if output.exists():
                lines = output.read_text().splitlines(keepends=True)
                complete = [line for line in lines if line.endswith("\n")]
                for line in complete[printed:]:
                    json.loads(line)  # fail instead of transporting a corrupt record
                    print("B0_RECORD " + base64.b64encode(line.encode()).decode(), flush=True)
                    printed += 1
            if time.monotonic() - last_status >= 15 or rc is not None:
                tail = logpath.read_text(errors="replace")[-500:]
                print("B0_STATUS " + json.dumps({"rows": printed, "rc": rc, "tail": tail}), flush=True)
                last_status = time.monotonic()
            if rc is not None:
                if rc != 0:
                    raise RuntimeError(f"generation exited {rc}; {printed} records streamed")
                return printed
            time.sleep(1)


if __name__ == "__main__":
    main()
