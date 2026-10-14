"""Recompute HN00-HN06 fixed-load results and verify persistent evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read_json(out / "protocol.json")
    freeze_path = out / "final-execution-freeze.json"
    if freeze_path.exists():
        freeze = read_json(freeze_path)
        native = Path(protocol["native_source"])
        build = Path(protocol["build"]) / "bin"
        for name, digest in freeze["source_sha256"].items():
            if hashlib.sha256((native / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Final source changed: {name}")
        for name, digest in freeze["binary_sha256"].items():
            if hashlib.sha256((build / name).read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Final binary changed: {name}")
    for path, digest in protocol["protected_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Protected asset changed: {path}")
    source = Path(protocol["source_checkpoint"])
    for name, identity in protocol["source_shard_identity"].items():
        st = (source / name).stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (identity["bytes"], identity["mtime_ns"], identity["inode"]):
            raise AssertionError(f"Checkpoint shard identity changed: {name}")
    shapes = [(128, 32), (2048, 32), (128, 256), (2048, 256)]
    rows = []
    records = {}
    selection = read_json(out / "candidate-selection.json")
    candidate_folder = selection["formal_candidate_folder"]
    for config, folder in [("cpu-provisional", "cpu-isolated"), ("original", "final-original"), ("candidate", candidate_folder)]:
        requests = read_json(out / folder / "requests.json")
        if len(requests) != 24 or sum(not r["warmup"] for r in requests) != 20:
            raise AssertionError(f"Incomplete formal campaign: {config}")
        guard = read_json(out / (folder + "-guard") / "summary.json")
        if config == "cpu-provisional":
            if not guard["passed"] or guard["own_gpu_limit_bytes"] != 0:
                raise AssertionError("CPU-only guard failed")
            gpu_peak = 0
        else:
            if guard["returncode"] or guard["error"]:
                raise AssertionError(f"GPU guard failed: {config}")
            gpu_peak = guard["gpu_peak_mib"]
        if gpu_peak > (7680 if config == "candidate" else 8192) or guard["rss_peak_gib"] > 64:
            raise AssertionError(f"Resource contract failed: {config}")
        records[config] = requests
        for length, output in shapes:
            samples = [r for r in requests if not r["warmup"] and (r["input_tokens"], r["output_tokens"]) == (length, output)]
            if len(samples) != 5 or len({tuple(r["output_ids"]) for r in samples}) != 1:
                raise AssertionError(f"Missing repeats or inconsistent output: {config}")
            for r in samples:
                work = r["final"]["s11_work"]
                expected = (length + output - 1) * 40 * 8
                if len(r["output_ids"]) != output or work["cpu_assignments"] + work["gpu_assignments"] != expected:
                    raise AssertionError(f"Work/token contract failed: {config}")
                if (config == "cpu-provisional" and work["gpu_assignments"]) or (config != "cpu-provisional" and work["cpu_assignments"]):
                    raise AssertionError(f"Unexpected routed execution placement: {config}")
            rows.append(dict(config=config, input_tokens=length, output_tokens=output,
                             wall_p50_s=statistics.median(r["wall_s"] for r in samples),
                             ttft_p50_s=statistics.median(r["ttft_s"] for r in samples),
                             decode_p50_tps=statistics.median(r["decode_tps"] for r in samples),
                             gpu_peak_mib=gpu_peak, rss_peak_gib=guard["rss_peak_gib"]))
    if any(a["output_ids"] != b["output_ids"] for a, b in zip(records["original"], records["candidate"])):
        raise AssertionError("Candidate changed frozen original output IDs")
    smoke = read_json(out / "launcher-smoke" / "requests.json")
    if len(smoke) != 2 or any(r["output_ids"] != records["candidate"][0]["output_ids"] for r in smoke):
        raise AssertionError("Persistent launcher changed selected output")
    for folder in ["head-qualification-guard", "forced-embedding-state-guard", "packed-state-guard"]:
        guard = read_json(out / folder / "summary.json")
        if guard["returncode"] or guard["error"]:
            raise AssertionError(f"Numerical/state qualification failed: {folder}")
    with (out / "hn-measurements.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rejected = read_json(out / "final-candidate" / "requests.json")
    if len(rejected) != 24 or sum(not r["warmup"] for r in rejected) != 20:
        raise AssertionError("Rejected candidate evidence was not retained")
    result = dict(passed=True, formal_samples=60, total_retained_formal_samples=80,
                  retained_rejected_formal_samples=20, launcher_smoke_passed=True,
                  all_gpu_output_ids_equal=True,
                  useful_work_exact=True, physical_8g_device_validated=False,
                  stable_b1_victory=False, cpu_comparison="provisional slow trunk, nonexclusive; not mature CPU or paired stability",
                  selected=selection["selected"],
                  rows=rows, scope="HN00-HN06 only; HN07 and later not executed")
    (out / "hn-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
