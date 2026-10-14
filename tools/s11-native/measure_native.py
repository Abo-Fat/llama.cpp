"""Common real-token SSE campaign for S11-Native/B1; execute under resource guard."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

import requests


def request(url, prompt, output, timeout=1800):
    started = time.perf_counter()
    events = []
    ids = []
    final = None
    with requests.post(
        url + "/completion",
        json=dict(prompt=prompt, n_predict=output, seed=0, temperature=0,
                  ignore_eos=True, stream=True, cache_prompt=False, return_tokens=True),
        stream=True, timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if final is not None:
                raise RuntimeError("SSE event after completion")
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            delta = event.get("tokens", [])
            if not isinstance(delta, list) or any(type(token) is not int for token in delta):
                raise ValueError("Invalid streamed token IDs")
            if delta:
                ids.extend(delta)
                events.append(dict(elapsed_s=time.perf_counter()-started, tokens=delta))
            if event.get("stop"):
                final = event
    wall = time.perf_counter() - started
    if len(ids) != output or len(events) < 2 or final is None:
        raise AssertionError("SSE incomplete or falsely buffered")
    if final["timings"]["prompt_n"] != len(prompt) or final["timings"]["predicted_n"] != output:
        raise AssertionError("Fixed input/output workload mismatch")
    decode = events[-1]["elapsed_s"] - events[0]["elapsed_s"]
    if decode <= 0:
        raise AssertionError("Invalid token timestamps")
    return dict(wall_s=wall, ttft_s=events[0]["elapsed_s"], decode_tps=(output-1)/decode,
                output_ids=ids, token_events=events, final=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18141)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--require-work", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.output.exists() or not command:
        parser.error("Require a new output directory and server command")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port already occupied")
    args.output.mkdir(parents=True)
    tokens = json.loads(args.prompt_tokens.read_text())
    if len(tokens) < 2048:
        raise ValueError("Frozen input shorter than formal workload")
    records = []
    url = f"http://127.0.0.1:{args.port}"
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
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
                time.sleep(.2)
            else:
                raise TimeoutError("Server not responsive before deadline")
            shapes = [(128, 32)] if args.smoke else [(128, 32), (2048, 32), (128, 256)] if args.screen else [(128, 32), (2048, 32), (128, 256), (2048, 256)]
            for length, output in shapes:
                previous = None
                for repeat in range(2 if args.smoke or args.screen else 6):
                    sample = request(url, tokens[:length], output, args.request_timeout)
                    if args.require_work:
                        work = sample["final"].get("s11_work")
                        if work is None or work["cpu_assignments"] + work["gpu_assignments"] != (length + output - 1) * 40 * 8:
                            raise AssertionError("Native natural routed workload mismatch")
                    if previous is not None and sample["output_ids"] != previous:
                        raise AssertionError("Repeated requests changed output IDs")
                    previous = sample["output_ids"]
                    sample.update(input_tokens=length, output_tokens=output,
                                  repeat=repeat, warmup=repeat == 0)
                    sample["graph_mode"] = "off" if any(v.startswith("GGML_CUDA_DISABLE_GRAPHS=") for v in command) else "default"
                    records.append(sample)
                    (args.output / "requests.json").write_text(json.dumps(records, indent=2))
                    print(json.dumps({k: sample[k] for k in ("input_tokens","output_tokens","repeat","wall_s","ttft_s","decode_tps")}), flush=True)
            (args.output / "measurement.json").write_text(json.dumps(
                dict(command=command, formal_samples=0 if args.smoke or args.screen else 20,
                     smoke=args.smoke, screening=args.screen, boundary="token-aware HTTP/SSE; empty text still counted"),
                indent=2,
            ))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
