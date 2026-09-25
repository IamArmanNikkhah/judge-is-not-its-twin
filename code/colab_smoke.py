#!/usr/bin/env python3
"""Authorized B0 four-output run using native colab run plus a deadline guard.

No retries, purchases, alternate GPUs, resume, scorer or judge. Capture stdout
continuously; native run cleans up and a separate process backs up that cleanup.
"""
import argparse
import base64
import hashlib
import json
import os
import queue
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / ".venv/bin/colab"


def capture_identity(session, out):
    identity = out / "identity.json"
    if identity.exists():
        try:
            return json.loads(identity.read_text())
        except ValueError:
            return None
    state_path = out / "sessions.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        record = state.get(session)
        if not record:
            return None
        value = {"session": session, "endpoint": record["endpoint"]}
        # Guard and supervisor may race; atomically publish the same observed ID.
        tmp = out / f"identity-{os.getpid()}.tmp"
        tmp.write_text(json.dumps(value))
        tmp.replace(identity)
        return value
    except (ValueError, KeyError):
        return None


def endpoint_check(endpoint=None):
    cmd = [sys.executable, str(ROOT / "code/colab_endpoint.py")]
    if endpoint:
        cmd += ["--stop-endpoint", endpoint]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return json.loads(result.stdout)


def stop(session, out):
    identity = capture_identity(session, out)
    try:
        r = subprocess.run([str(COLAB), "--config", str(out / "sessions.json"),
                            "stop", "-s", session], capture_output=True,
                           text=True, timeout=40)
        data = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except subprocess.TimeoutExpired:
        data = {"returncode": 124, "error": "stop client timed out; release unconfirmed"}
    data["verified_absent"] = False
    if identity:
        try:
            verification = endpoint_check(identity["endpoint"])
            data["verification"] = verification
            data["verified_absent"] = verification["target_absent"]
        except (RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            data["verification_error"] = str(exc)
    with (out / "stop-attempts.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), **data}) + "\n")
    return data


def guard(session, out, deadline):
    # Check parent completion once a second; never block the interactive agent.
    while time.monotonic() < deadline and not (out / "supervisor-finished").exists():
        capture_identity(session, out)
        time.sleep(1)
    if not (out / "supervisor-finished").exists():
        for _ in range(3):
            result = stop(session, out)
            print(json.dumps({"deadline_stop": result}), flush=True)
            if result["verified_absent"]:
                return


# The frozen 09-05 smoke settings are the defaults; the Phase B pilot passes its own
# (PHASE-B-protocol.md: --elicitation per arm, --max-new-tokens 768, 12 ff prompts, k 4).
SMOKE_SPEC = {"elicitation": "ordinary", "max_new_tokens": 512, "k": 2,
              "prompt_ids": ["ff-01", "ff-02"]}


def generation_spec(elicitation, max_new_tokens, k, prompt_ids, temperature=1.0, top_p=0.95, run_id=None, k_start=0, seed=411):
    if elicitation not in ("ordinary", "unlikely", "decoding"):
        raise ValueError(f"unknown elicitation {elicitation!r}")
    if max_new_tokens < 1 or k < 1 or not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("bad generation spec")
    if not (0 < temperature <= 3) or not (0 < top_p <= 1) or k_start < 0:
        raise ValueError("bad sampling spec")
    return {"elicitation": elicitation, "max_new_tokens": max_new_tokens, "k": k,
            "prompt_ids": list(prompt_ids), "expected_records": k * len(prompt_ids),
            "temperature": temperature, "top_p": top_p, "run_id": run_id, "k_start": k_start, "seed": seed}


def select_prompts(rows, prompt_ids):
    by_id = {p["id"]: p for p in rows}
    missing = [i for i in prompt_ids if i not in by_id]
    if missing:
        raise ValueError(f"prompt IDs not in prompts.jsonl: {missing}")
    return [by_id[i] for i in prompt_ids]


def expected_item_id(run_id, pid, elicitation, index, seed=411):
    # Mirrors generate.item_id_for: the ordinary arm keeps the Phase A shape, the other arm is tagged.
    arm = "" if elicitation == "ordinary" else f"__{elicitation}"
    return f"{run_id}__{pid}__zephyr-dpo{arm}__s{seed}k{index}"


def validate_record(row, run_id, seen, spec=SMOKE_SPEC):
    expected = {"source_model": "HuggingFaceH4/zephyr-7b-beta", "lineage": "zephyr",
                "stage": "dpo", "run_id": spec.get("run_id") or run_id, "frame": "completion"}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError("record source/frame/run mismatch")
    if row.get("elicitation", "ordinary") != spec["elicitation"]:
        raise ValueError("record elicitation arm mismatch")
    params = row["params"]
    if any(params.get(k) != v for k, v in {"max_new_tokens": spec["max_new_tokens"],
                                          "temperature": spec.get("temperature", 1.0),
                                          "top_p": spec.get("top_p", .95), "seed": spec.get("seed", 411)}.items()):
        raise ValueError("record sampling settings mismatch")
    pid, index = row["prompt_id"], params["k_index"]
    if pid not in spec["prompt_ids"] or not spec.get("k_start", 0) <= index < spec.get("k_start", 0) + spec["k"]:
        raise ValueError("unexpected generation slot")
    ident = expected_item_id(spec.get("run_id") or run_id, pid, spec["elicitation"], index, spec.get("seed", 411))
    if row["item_id"] != ident or ident in seen:
        raise ValueError("duplicate/wrong item ID")
    if hashlib.sha1(row["text"].encode()).hexdigest()[:12] != row["sha1"]:
        raise ValueError("text hash mismatch")
    term = row["termination"]
    if not 0 < term["generated_token_count"] <= spec["max_new_tokens"]:
        raise ValueError("invalid generated token count")
    seen.add(ident)


def validate_score_record(row, job, seen):
    """typicality.py rows for the manipulation check: one per kept item, one reader, prefix frame,
    never truncated, per-token array present and consistent with the token count."""
    ident = row["item_id"]
    if ident not in job["item_ids"] or ident in seen:
        raise ValueError("unexpected/duplicate scored item")
    if row.get("ref_tag") != job["ref_tag"] or row.get("ref_model") != job["ref_model"]:
        raise ValueError("reader mismatch")
    if row.get("frame") != "completion-prefix" or row.get("max_len") != job["max_len"]:
        raise ValueError("frame/max_len mismatch")
    if row.get("truncated") is not False:
        raise ValueError("truncated score banked; Rule 2 forbids it")
    nll, n_tok, per = row.get("mean_nll"), row.get("n_tokens"), row.get("token_nll")
    if not isinstance(nll, (int, float)) or isinstance(nll, bool) or nll != nll or nll < 0:
        raise ValueError("bad mean_nll")
    if not isinstance(n_tok, int) or n_tok < 1 or not isinstance(per, list) or len(per) != n_tok:
        raise ValueError("token_nll array inconsistent with n_tokens")
    seen.add(ident)


def judge_job(pairs, judge_model, lineage, stage, criterion, max_len):
    """Competence-check pairs under one judge. Frame 'completion' and max_len 1024 with no
    refuse-on-overflow reproduce the Phase A gate; only the criterion wording may differ."""
    keys = [[p["item_x"], p["item_y"]] for p in pairs]
    if not keys or len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("pairs must be nonempty and unique")
    return {"kind": "judge", "judge_model": judge_model, "judge_lineage": lineage, "judge_stage": stage,
            "criterion": criterion, "max_len": max_len, "pair_keys": keys, "expected_records": len(keys)}


def record_key(row):
    """Printable id for any banked row: stories/scores carry item_id, judged pairs carry item_x/item_y."""
    return row.get("item_id") or f"{row.get('item_x')}|{row.get('item_y')}"


def validate_judge_record(row, job, seen):
    key = (row.get("item_x"), row.get("item_y"))
    if list(key) not in job["pair_keys"] or key in seen:
        raise ValueError("unexpected/duplicate judged pair")
    for f in ("judge_model", "judge_lineage", "judge_stage", "criterion", "max_len"):
        if row.get(f) != job[f]:
            raise ValueError(f"{f} mismatch")
    if row.get("frame") != "completion":
        raise ValueError("frame mismatch")
    for f in ("p_x", "p_x_first_order", "p_y_first_order"):
        v = row.get(f)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            raise ValueError(f"bad {f}")
    seen.add(key)


def scoring_job(items, ref_tag, ref_model, max_len):
    ids = [i["item_id"] for i in items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("items must be nonempty with unique item_id")
    return {"kind": "typicality", "ref_tag": ref_tag, "ref_model": ref_model, "max_len": max_len,
            "item_ids": ids, "expected_records": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="b0-completion-smoke-20260905")
    ap.add_argument("--guard-deadline", type=float)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--elicitation", choices=["ordinary", "unlikely", "decoding"], default=SMOKE_SPEC["elicitation"],
                    help="Phase B arm; one arm per session, each with its own --run-id and --out-dir")
    ap.add_argument("--max-new-tokens", type=int, default=SMOKE_SPEC["max_new_tokens"])
    ap.add_argument("--k", type=int, default=SMOKE_SPEC["k"], help="samples per prompt")
    ap.add_argument("--k-start", type=int, default=0, help="first k_index; top-up runs fill later slots under the same --run-id")
    ap.add_argument("--seed", type=int, default=411,
                    help="torch seed. A top-up MUST use a fresh seed: same seed + same prompt order = identical stories (09-06 k3 session duplicated k2 12/12)")
    ap.add_argument("--prompt-ids", default=",".join(SMOKE_SPEC["prompt_ids"]),
                    help="comma-separated prompt IDs from data/items/prompts.jsonl, in run order")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sweep-temperatures", default=None,
                    help="comma list; one generation spec per dose in ONE session, run_id gets -t<T>")
    ap.add_argument("--budget-seconds", type=int, default=1800,
                    help="hard session budget; the guard fires 300 s before it")
    ap.add_argument("--job", choices=["generate", "typicality", "judge"], default="generate",
                    help="typicality = score --items under --ref-model (manipulation check); generation flags ignored")
    ap.add_argument("--items", help="typicality: JSONL of kept stories (item_id, prompt_id, text)")
    ap.add_argument("--ref-tag", default="olmo2-base")
    ap.add_argument("--ref-model", default="allenai/OLMo-2-1124-7B")
    ap.add_argument("--scorer-max-len", type=int, default=1024, help="typicality: protocol Rule 2 item cap")
    ap.add_argument("--pairs", help="judge: JSONL pairs file (prompt_id, item_x, item_y, ...); --items holds their texts")
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-lineage")
    ap.add_argument("--judge-stage")
    ap.add_argument("--criterion", default="better-grammatical")
    ap.add_argument("--judge-max-len", type=int, default=1024, help="judge: 1024 reproduces Phase A")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    if args.guard_deadline is not None:
        guard(args.session, out, args.guard_deadline)
        return
    if out.exists():
        raise ValueError(f"refusing to overwrite run directory {out}")
    if args.budget_seconds < 600:
        raise ValueError("budget below 600 s leaves no room for weight download")
    all_prompts = [json.loads(x) for x in (ROOT / "data/items/prompts.jsonl").read_text().splitlines()]
    if args.job == "judge":
        if not (args.pairs and args.items and args.judge_model and args.judge_lineage and args.judge_stage):
            raise ValueError("--job judge needs --pairs --items --judge-model --judge-lineage --judge-stage")
        pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
        need = {p["item_x"] for p in pairs} | {p["item_y"] for p in pairs}
        items = [r for r in (json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip())
                 if r["item_id"] in need]
        if {i["item_id"] for i in items} != need:
            raise ValueError("items file is missing texts for some pairs")
        spec = judge_job(pairs, args.judge_model, args.judge_lineage, args.judge_stage, args.criterion, args.judge_max_len)
        prompts = select_prompts(all_prompts, sorted({p["prompt_id"] for p in pairs}))
        source = (ROOT / "code/judge.py").read_text()
        bundle = {"run_id": args.run_id, "model": args.judge_model, "script_name": "judge.py",
                  "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "prompts": prompts, "items": items, "pairs": pairs, "job": spec}
        validate = lambda row, seen: validate_judge_record(row, spec, seen)
        progress = lambda row: f"p_x={row['p_x']} {row.get('ruin', '')}"
    elif args.job == "typicality":
        if not args.items:
            raise ValueError("--job typicality needs --items")
        items = [{k: r[k] for k in ("item_id", "prompt_id", "text")}
                 for r in (json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip())]
        spec = scoring_job(items, args.ref_tag, args.ref_model, args.scorer_max_len)
        prompts = select_prompts(all_prompts, sorted({i["prompt_id"] for i in items}))
        source = (ROOT / "code/typicality.py").read_text()
        bundle = {"run_id": args.run_id, "model": args.ref_model, "script_name": "typicality.py",
                  "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "prompts": prompts, "items": items, "job": spec}
        validate = lambda row, seen: validate_score_record(row, spec, seen)
        progress = lambda row: f"nll={row['mean_nll']} n={row['n_tokens']}"
    else:
        ids = args.prompt_ids.split(",")
        if args.sweep_temperatures:
            specs = [generation_spec(args.elicitation, args.max_new_tokens, args.k, ids, float(t), args.top_p,
                                     run_id=f"{args.run_id}-t{t}") for t in args.sweep_temperatures.split(",")]
        else:
            specs = [generation_spec(args.elicitation, args.max_new_tokens, args.k, ids, args.temperature, args.top_p, k_start=args.k_start, seed=args.seed)]
        by_run = {s["run_id"] or args.run_id: s for s in specs}
        spec = {"expected_records": sum(s["expected_records"] for s in specs), "doses": specs}
        prompts = select_prompts(all_prompts, ids)
        source = (ROOT / "code/generate.py").read_text()
        bundle = {"run_id": args.run_id, "model": "HuggingFaceH4/zephyr-7b-beta",
                  "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "prompts": prompts}
        if args.sweep_temperatures:
            bundle["generations"] = specs
        else:
            bundle["generation"] = specs[0]
        def validate(row, seen):
            s = by_run.get(row.get("run_id"))
            if s is None:
                raise ValueError(f"record run_id {row.get('run_id')!r} matches no dose in this session")
            validate_record(row, args.run_id, seen, s)
        progress = lambda row: f"T={row['params']['temperature']} {row['termination']}"
    budget, soft = args.budget_seconds, args.budget_seconds - 300
    out.mkdir(parents=True)
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    payload = ROOT / "code/colab_smoke_payload.py"
    (out / "code-hashes.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in (Path(__file__), payload)}, indent=2) + "\n")
    if args.prepare_only:
        print(f"PREPARED ONLY {out}; no Colab call. spec={json.dumps(spec)} budget={budget}s")
        return
    before = endpoint_check()
    (out / "before-sessions.json").write_text(json.dumps(before) + "\n")
    started = time.monotonic()
    (out / "started.json").write_text(json.dumps({"session": args.session, "wall_time": time.time(),
                                                "monotonic": started, "hard_budget_seconds": budget,
                                                "generation": spec}) + "\n")
    # 300 s before the budget the guard requests release, leaving room for API latency.
    with (out / "guard.log").open("w") as guard_log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--session", args.session,
                          "--out-dir", str(out), "--guard-deadline", str(started + soft)],
                         stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
    # --keep avoids native teardown's swallowed errors/state deletion; our
    # explicit stop uses the banked endpoint and verifies server absence.
    command = [str(COLAB), "--config", str(out / "sessions.json"),
               "run", "--keep", "--gpu", "T4", "-s", args.session, "--timeout", str(soft),
               str(payload), base64.b64encode(json.dumps(bundle).encode()).decode()]
    seen, proc = set(), None
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        messages = queue.Queue()
        def reader():
            for line in proc.stdout:
                messages.put(line)
            messages.put(None)
        threading.Thread(target=reader, daemon=True).start()
        with (out / "client.log").open("w") as log, (out / "outputs.jsonl").open("x") as bank:
            while time.monotonic() - started < soft:
                capture_identity(args.session, out)
                try:
                    line = messages.get(timeout=1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if "B0_RECORD " in line:
                    encoded = line.split("B0_RECORD ", 1)[1].strip()
                    row = json.loads(base64.b64decode(encoded))
                    validate(row, seen)
                    bank.write(json.dumps(row) + "\n")
                    bank.flush()
                    os.fsync(bank.fileno())
                    print(f"BANKED {len(seen)}/{spec['expected_records']} {record_key(row)} {progress(row)}", flush=True)
                else:
                    print(line, end="", flush=True)
            else:
                print(f"{soft}-second execution boundary reached; stopping and keeping partials", flush=True)
                proc.terminate()
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=5)
        print(f"NATIVE_RUN_EXIT {rc}; banked={len(seen)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    finally:
        # Native run normally releases first; explicit stop handles errors too.
        for _ in range(3):
            result = stop(args.session, out)
            if result["verified_absent"]:
                break
        print("FINAL_STOP " + json.dumps(result), flush=True)
        if result["verified_absent"]:
            (out / "supervisor-finished").write_text("target absent from native server assignment list\n")
        if proc is not None and proc.poll() is None:
            proc.terminate()
    if rc != 0 or len(seen) != spec["expected_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
