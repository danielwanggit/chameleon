"""Property test: random variants of the sample designs x every catalogue
size must never crash and must pass every hard rule. Reports how often the
safe fallback layout had to be used (a quality signal, not a failure).

    python -m eval.stress [N_VARIANTS] [SEED]
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

from engine import resize as R
from eval.catalogue import CATALOGUE

ROOT = Path(__file__).resolve().parent.parent
WORDS = "golden hour botanical sparkling infusion citrus summer light crafted small batch limited fresh".split()


def variant(src: dict, rng: random.Random) -> dict:
    d = copy.deepcopy(src)
    els = d["elements"]
    W, H = d["canvas"]["width"], d["canvas"]["height"]
    # random source canvas from the catalogue (rescale geometry to it)
    _, nw, nh = rng.choice(CATALOGUE)
    if rng.random() < 0.5:  # same design authored at another size (uniform, centred)
        k = min(nw / W, nh / H)
        ox, oy = (nw - W * k) / 2, (nh - H * k) / 2
        for e in els:
            if e["type"] == "image" and e.get("fit") == "cover":
                e.update(x=0, y=0, width=nw, height=nh)
                continue
            e["x"] = ox + e["x"] * k
            e["y"] = oy + e["y"] * k
            e["width"] *= k
            e["height"] *= k
            if e["type"] == "text":
                e["font_size"] = max(6, round(e["font_size"] * k))
        d["canvas"] = {"width": nw, "height": nh}
    for e in els:
        if e["type"] == "text" and rng.random() < 0.3:  # awkward copy lengths
            n = rng.choice([1, 2, 8, 25])
            e["content"] = " ".join(rng.choice(WORDS) for _ in range(n)).title()
            e.pop("style_runs", None)
        if e["type"] == "text" and rng.random() < 0.15:
            e["text_align"] = rng.choice(["left", "center", "right"])
        if rng.random() < 0.15 and not e.get("group_id"):  # jitter positions
            e["x"] += rng.uniform(-0.1, 0.1) * d["canvas"]["width"]
            e["y"] += rng.uniform(-0.1, 0.1) * d["canvas"]["height"]
    # occasionally drop an ungrouped element (variable element counts)
    if rng.random() < 0.3:
        cands = [e for e in els if not e.get("group_id") and e["type"] != "image"]
        if len(cands) > 1:
            els.remove(rng.choice(cands))
    return d


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    sources = [json.loads(p.read_text())["source"] for p in sorted((ROOT / "input").glob("*.json"))]
    cases = fails = fallbacks = crashes = 0
    t0 = time.time()
    for i in range(n):
        src = variant(rng.choice(sources), rng)
        for name, w, h in CATALOGUE:
            target = {"width": w, "height": h}
            cases += 1
            try:
                out = R.resize(src, target)
            except Exception as exc:  # resize() must never raise
                crashes += 1
                print(f"CRASH variant {i} @ {name}: {exc!r}")
                continue
            bad = R.check(src, out, target)["rule"]
            if bad:
                fails += 1
                print(f"RULE variant {i} @ {name}: {bad[:3]}")
            if out == R.safe_layout(src, target):
                fallbacks += 1
    dt = time.time() - t0
    print(f"{cases} cases in {dt:.1f}s ({1000 * dt / cases:.0f} ms/case): "
          f"{crashes} crashes, {fails} hard-rule failures, {fallbacks} safe-layout fallbacks")
    return 1 if crashes or fails else 0


if __name__ == "__main__":
    sys.exit(main())
