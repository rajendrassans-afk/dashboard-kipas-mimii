"""Pembuat data SINTETIS (hanya untuk menguji aplikasi; BUKAN pengganti MIMII).
Struktur sama dengan MIMII: <out>/fan/id_XX/{normal,abnormal}/*.wav (16 kHz, 10 s)."""
import sys, argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np, soundfile as sf

from core.synth import fan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="demo_synthetic")
    ap.add_argument("--n_normal", type=int, default=120); ap.add_argument("--n_abn", type=int, default=50)
    ap.add_argument("--ids", type=int, default=2); a = ap.parse_args()
    rng = np.random.default_rng(1)
    for i in range(a.ids):
        f0 = 45 + 6 * i
        for lab, n in (("normal", a.n_normal), ("abnormal", a.n_abn)):
            d = Path(a.out) / "fan" / f"id_{i*2:02d}" / lab; d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                kind = "normal" if lab == "normal" else rng.choice(["imbalance", "rattle", "friction"])
                y = fan(rng, kind, f0=f0 + rng.normal(0, .5))
                sf.write(d / f"{k:08d}.wav", y, 16000)
    print("selesai ->", a.out)
main()
