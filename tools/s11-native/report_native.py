"""Verify and summarize the final S11-Native/B1 common-request campaign."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    root = Path("/home/v-zning/workspace_qwen")
    freeze = json.loads((run / "native-execution-freeze.json").read_text())
    cpu_build_path = run / "cpu-production-build.json"
    cpu_build = json.loads(cpu_build_path.read_text()) if cpu_build_path.exists() else {}
    for path, expected in freeze["source_sha256"].items():
        actual = hashlib.sha256((root / "s11-native" / path).read_bytes()).hexdigest()
        if actual != expected:
            correction = cpu_build.get("source_updates", {}).get(path)
            if correction is None or correction["measured_sha256"] != expected or correction["current_sha256"] != actual:
                raise AssertionError(f"Unrecorded native execution source change: {path}")
    if hashlib.sha256((root / "baseline-envs/s11-native/build/bin/llama-server").read_bytes()).hexdigest() != freeze["binary_sha256"]:
        raise AssertionError("Native binary changed during formal measurement")
    protocol = json.loads((run / "protocol.json").read_text())
    for path, expected in protocol["protected_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Protected checkpoint/B1 asset changed: {path}")
    shapes = ((128, 32), (2048, 32), (128, 256), (2048, 256))
    configurations = ("native", "native-graph-off", "b1")
    data = {}
    rows = []
    for name in configurations:
        directory = run / f"formal-{name}-v2"
        data[name] = json.loads((directory / "requests.json").read_text())
        metadata = json.loads((directory / "measurement.json").read_text())
        guard = json.loads((run / f"formal-{name}-v2-guard/summary.json").read_text())
        if metadata["formal_samples"] != 20 or guard["returncode"] != 0 or guard["error"]:
            raise AssertionError(f"Missing valid formal campaign: {name}")
        for shape in shapes:
            cell = [s for s in data[name] if not s["warmup"] and (s["input_tokens"], s["output_tokens"]) == shape]
            if len(cell) != 5 or any(len(s["output_ids"]) != shape[1] for s in cell):
                raise AssertionError(f"Missing exact native workload: {name} {shape}")
            if any(s["output_ids"] != cell[0]["output_ids"] for s in cell):
                raise AssertionError("Repeated-request output/state changed")
            work = [s["final"].get("s11_work") for s in cell]
            row = dict(config=name, input_tokens=shape[0], output_tokens=shape[1],
                       repeats=5, wall_p50_s=median(s["wall_s"] for s in cell),
                       ttft_p50_s=median(s["ttft_s"] for s in cell),
                       decode_p50_tps=median(s["decode_tps"] for s in cell),
                       wall_min_s=min(s["wall_s"] for s in cell),
                       wall_max_s=max(s["wall_s"] for s in cell),
                       gpu_peak_mib=guard["gpu_peak_mib"], rss_peak_gib=guard["rss_peak_gib"])
            if name != "b1":
                expected = (shape[0] + shape[1] - 1) * 40 * 8
                for value in work:
                    if value is None or value["cpu_assignments"] + value["gpu_assignments"] != expected:
                        raise AssertionError("Native actual routed tasks changed")
                    if (name == "native-graph-off") != (value["graph_replays"] == 0):
                        raise AssertionError("Native Graph on/off did not really execute")
                row.update(cpu_assignments=median(w["cpu_assignments"] for w in work),
                           gpu_assignments=median(w["gpu_assignments"] for w in work),
                           graph_replays=median(w["graph_replays"] for w in work),
                           routed_h2d_mib=median(w["routed_h2d_bytes"] for w in work)/2**20,
                           route_readbacks=median(w["route_readbacks"] for w in work),
                           route_syncs=median(w["route_syncs"] for w in work))
            rows.append(row)
    for on, off in zip(data["native"], data["native-graph-off"], strict=True):
        if on["output_ids"] != off["output_ids"]:
            raise AssertionError("Graph-only optimization changed model output")
        a, b = on["final"]["s11_work"], off["final"]["s11_work"]
        for key in ("cpu_assignments", "gpu_assignments"):
            if a[key] != b[key]:
                raise AssertionError("Graph-only optimization changed useful work")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (run / "measurements.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    successful = []
    exclusions = []
    for path in run.glob("*-guard/summary.json"):
        guard = json.loads(path.read_text())
        if guard["error"] or guard["returncode"]:
            exclusions.append(dict(run=path.parent.name, error=guard["error"]))
            continue
        samples = list(csv.DictReader((path.parent / "resources.csv").open()))
        gpu = max(int(s["gpu_bytes"]) for s in samples)/2**20
        rss = max(int(s["process_group_rss_bytes"]) for s in samples)/2**30
        if gpu != guard["gpu_peak_mib"] or rss != guard["rss_peak_gib"] or gpu > 8192 or rss > 64:
            raise AssertionError(f"Resource raw evidence disagrees: {path}")
        successful.append(path.parent.name)
    evidence = [
        run / "protocol.json", run / "native-execution-freeze.json",
        run / "source-to-native-manifest.json", run / "roundtrip-validation.json",
        run / "container-native-payload-validation.json", run / "container-loader-validation.json",
        run / "container-type-validation.json", run / "cpu-qualification.json",
        run / "measurements.csv", run / "policy-selection.json",
        run / "cpu-production-build.json", run / "cpu-production-qualification.json",
        *run.glob("*-guard/summary.json"), *run.glob("*-guard/resources.csv"),
        *run.glob("formal-*/requests.json"), *run.glob("dev-*/requests.json"),
    ]
    (run / "verification.json").write_text(json.dumps(dict(
        passed=True, final_formal_samples=60, earlier_retained_formal_samples=60,
        measured_gpu_execution_verified=True, cpu_production_correction=cpu_build,
        protected_assets_unchanged=True,
        on_off_outputs_and_useful_work_equal=True, successful_guards=successful,
        excluded_diagnostics=exclusions,
        evidence_sha256={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        stable_baseline_victory=False,
        conclusion="Verified basic native implementation, not a claim of beating B1 or completing three-round victory acceptance.",
    ), indent=2))
    for row in rows:
        print(row["config"], f'{row["input_tokens"]}/{row["output_tokens"]}',
              *(round(row[k],4) for k in ("wall_p50_s","ttft_p50_s","decode_p50_tps")),
              {k: row[k] for k in ("routed_h2d_mib","route_readbacks","route_syncs") if k in row})
    print("PASS: final 60 complete requests, Graph causality, work/resource/source evidence.")


if __name__ == "__main__":
    main()
