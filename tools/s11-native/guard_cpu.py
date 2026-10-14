"""CPU-only experiment guard; foreign GPU activity is recorded, not interrupted."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
import pynvml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=64800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.output.exists():
        parser.error("Require a new output directory, command and positive timeout")
    args.output.mkdir(parents=True)
    os.sched_setaffinity(0, range(8))
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    result = dict(command=command, passed=False, cpu_affinity=list(range(8)),
                  rss_limit_gib=64, own_gpu_limit_bytes=0,
                  foreign_policy="record only; provisional CPU control, not paired stability")
    started = time.monotonic()
    peak = 0
    foreign_seen = set()
    with (args.output / "command.log").open("w") as log, (args.output / "resources.csv").open("w") as resource:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        parent = psutil.Process(process.pid)
        writer = csv.writer(resource)
        writer.writerow(["elapsed_s", "group_rss_bytes", "own_gpu_bytes", "foreign_gpu_pids", "host_load1", "host_steal_s"])
        try:
            while process.poll() is None:
                group = [parent, *parent.children(recursive=True)]
                owned = {p.pid for p in group}
                rss = 0
                for child in group:
                    try:
                        rss += child.memory_info().rss
                        if not set(child.cpu_affinity()).issubset(range(8)):
                            raise RuntimeError("CPU-only child escaped 8-CPU affinity")
                    except psutil.NoSuchProcess:
                        continue
                active = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
                own = [p for p in active if p.pid in owned]
                foreign = [p.pid for p in active if p.pid not in owned]
                foreign_seen.update(foreign)
                used = sum(p.usedGpuMemory for p in own)
                peak = max(peak, rss)
                writer.writerow([time.monotonic()-started, rss, used, ";".join(map(str, foreign)),
                                 os.getloadavg()[0], psutil.cpu_times().steal])
                resource.flush()
                if own or rss > 64 * 2**30:
                    raise RuntimeError(f"CPU-only budget violated: GPU holders={len(own)}, RSS={rss}")
                if time.monotonic()-started > args.timeout:
                    raise TimeoutError("CPU-only experiment timed out")
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError(f"CPU-only command exited {process.returncode}")
            result["passed"] = True
        except (RuntimeError, TimeoutError) as exc:
            result["error"] = str(exc)
            raise
        finally:
            children = parent.children(recursive=True) if parent.is_running() else []
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    continue
            _, alive = psutil.wait_procs(children, timeout=10)
            for child in alive:
                child.kill()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            result.update(wall_s=time.monotonic()-started, rss_peak_gib=peak/2**30,
                          foreign_gpu_pids=sorted(foreign_seen), returncode=process.returncode)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
