import csv
import time
import argparse
import requests
import random
import math
from datetime import datetime

DEFAULT_DEVICE_URL = "http://192.168.4.1/data"

def sim_sample(t: float):
    """Simulated 4 temps + 8 pressures (similar to ESP32 SIM)."""
    temps = [
        34.0 + 0.2 * math.sin(t / 20.0) + 0.05 * i + 0.1 * math.sin((t + i * 3) / 7.0)
        for i in range(4)
    ]
    press = []
    for i in range(8):
        base = 1400 + 120 * math.sin(t / 8.0)
        per = 30 * i + 40 * math.sin((t + i * 2) / 5.0)
        noise = 20 * math.sin((t + i) * 1.7)
        v = base + per + noise + random.uniform(-5, 5)
        v = max(0, min(4095, v))
        press.append(int(v))
    return temps, press

def fetch_device_sample(url: str):
    """Pull one JSON sample from ESP32."""
    r = requests.get(url, timeout=2)
    r.raise_for_status()
    j = r.json()

    ts_ms = j.get("ts_ms")
    temps = j.get("temps_c", [None] * 4)
    press = j.get("press_raw", [None] * 8)

    # Force exact lengths
    temps = (temps + [None] * 4)[:4]
    press = (press + [None] * 8)[:8]
    return ts_ms, temps, press

def main():
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--mode", choices=["sim", "device"], default="sim")
        parser.add_argument("--url", default=DEFAULT_DEVICE_URL)
        parser.add_argument("--hz", type=float, default=2.0)
        parser.add_argument("--out", default="", help="output CSV filename (optional)")
        parser.add_argument("--tag", default="untagged",
                            help="label for this session, e.g. rest/walk/loose")
        return parser.parse_args()

    args = parse_args()
    interval = 1.0 / max(0.1, args.hz)

    out_csv = args.out.strip()
    if not out_csv:
        out_csv = f"bra_raw_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Mode: {args.mode}")
    if args.mode == "device":
        print(f"Device URL: {args.url}")
    print(f"Sampling: {args.hz} Hz (every {interval:.3f}s)")
    print(f"Writing: {out_csv}")
    print("Press Ctrl+C to stop.\n")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        # Header: simple + consistent
        w.writerow([
            "pc_time_iso", "ts_ms",
            "t1_c", "t2_c", "t3_c", "t4_c",
            "p1_raw", "p2_raw", "p3_raw", "p4_raw", "p5_raw", "p6_raw", "p7_raw", "p8_raw","tag"
        ])

        start = time.time()
        n = 0

        try:
            while True:
                pc_time = datetime.now().isoformat(timespec="milliseconds")

                if args.mode == "sim":
                    t = time.time() - start
                    temps, press = sim_sample(t)
                    ts_ms = int(t * 1000)
                else:
                    ts_ms, temps, press = fetch_device_sample(args.url)

                row = [pc_time, ts_ms] + temps + press + [args.tag]
                w.writerow(row)
                f.flush()

                n += 1
                if n % int(max(1, args.hz)) == 0:
                    print("logged", n, "samples")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopped. Saved:", out_csv)

if __name__ == "__main__":
    main()