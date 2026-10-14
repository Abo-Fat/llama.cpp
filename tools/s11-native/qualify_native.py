"""Native S11 loader/stream qualification; GPU runs additionally require run_guarded."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import psutil
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--expert-mode", choices=["cpu", "gpu-cache", "hybrid"], default="cpu")
    parser.add_argument("--input-tokens", type=int, choices=[128, 2048], default=128)
    parser.add_argument("--output-tokens", type=int, choices=[32, 256], default=32)
    args = parser.parse_args()
    root = Path("/home/v-zning/workspace_qwen")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    port = 18141
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("Native qualification port is occupied")
    command = [
        "taskset", "-c", "0-7", str(root / "baseline-envs/s11-native/build/bin/llama-server"),
        "-m", str(root / "baseline-models/s11-native/s11-native.gguf"),
        "--host", "127.0.0.1", "--port", str(port), "-ngl", "99" if args.gpu else "0",
        "--cpu-moe", "-t", "8", "-tb", "8", "-c", "4096", "-b", "512", "-ub", "128",
        "-np", "1", "--no-warmup", "--fit", "off", "-ctk", "bf16", "-ctv", "bf16",
        "--poll", "0", "--poll-batch", "0",
        "-fa", "off",
        "--backend-sampling",
    ]
    if not args.gpu:
        command.extend(["--device", "none"])
    if args.expert_mode != "cpu":
        if not args.gpu:
            parser.error("GPU cache mode requires GPU execution")
        command.extend(["--moe-cache-mib", "3072"])
    env = dict(os.environ, S11_EXPERT_MODE=args.expert_mode)
    if not args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    samples = []
    stop = threading.Event()
    errors = []
    result = dict(command=command, role="Native diagnostic, not formal performance",
                  gpu=args.gpu, expert_mode=args.expert_mode, passed=False)
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        parent = psutil.Process(process.pid)

        def monitor():
            while not stop.wait(0.05):
                if process.poll() is not None:
                    return
                try:
                    group = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    return
                rss = 0
                for child in group:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                samples.append(rss)
                if rss > 64 * 2**30:
                    errors.append("Native CPU diagnostic exceeded group RSS")
                    process.terminate()
                    return

        thread = threading.Thread(target=monitor)
        thread.start()
        try:
            deadline = time.monotonic() + 600
            url = f"http://127.0.0.1:{port}"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Native server exited {process.returncode}")
                try:
                    response = requests.get(url + "/health", timeout=1)
                    if response.status_code == 200:
                        break
                    if response.status_code != 503:
                        response.raise_for_status()
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Native server startup deadline exceeded")
            records = []
            if args.generate:
                prompt = json.loads((root / "outputs/baselines/20261009-abc/b3-exl3-cache0-fixed/prompt-tokens.json").read_text())[:args.input_tokens]
                for _ in range(2):
                    started = time.perf_counter()
                    ids = []
                    events = []
                    final = None
                    with requests.post(
                        url + "/completion", json=dict(
                            prompt=prompt, n_predict=args.output_tokens, temperature=0, seed=0,
                            ignore_eos=True, cache_prompt=False, stream=True, return_tokens=True,
                        ), stream=True, timeout=1200,
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines(chunk_size=1):
                            if not line.startswith(b"data: "):
                                continue
                            event = json.loads(line[6:])
                            if "error" in event:
                                raise RuntimeError(str(event["error"]))
                            delta = event.get("tokens", [])
                            if delta:
                                ids.extend(delta)
                                events.append(time.perf_counter() - started)
                            if event.get("stop"):
                                final = event
                    if len(ids) != args.output_tokens or len(events) < 2 or final is None:
                        raise AssertionError("Native output was incomplete or not really streamed")
                    if final["timings"]["prompt_n"] != args.input_tokens or final["timings"]["predicted_n"] != args.output_tokens:
                        raise AssertionError("Native workload count mismatch")
                    work = final.get("s11_work")
                    if work is None or work["cpu_assignments"] + work["gpu_assignments"] != (args.input_tokens + args.output_tokens - 1) * 40 * 8:
                        raise AssertionError("Native real expert workload mismatch")
                    records.append(dict(output_ids=ids, wall_s=time.perf_counter()-started, token_events=events, final=final))
                    if len(records) > 1 and records[-1]["output_ids"] != records[0]["output_ids"]:
                        raise AssertionError("Repeated native requests changed IDs")
                (args.output / "requests.json").write_text(json.dumps(records, indent=2))
            if errors:
                raise RuntimeError(errors[0])
            result["passed"] = True
            result["complete_requests"] = len(records)
        except (RuntimeError, TimeoutError, AssertionError, ValueError, requests.RequestException) as exc:
            result["error"] = str(exc)
            raise
        finally:
            stop.set()
            thread.join()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            result["rss_peak_bytes"] = max(samples, default=0)
            result["returncode_on_cleanup"] = process.returncode
            (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
