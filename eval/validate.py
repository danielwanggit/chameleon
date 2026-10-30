"""Submission validator: every output must survive the checks the scorer
may run, not only our own. For each case it checks:

  1. resize() returns (never raises) and is deterministic
  2. the output validates against chameleon_design_schema_v0.9.json
  3. engine/main.py's own self_check() reports nothing
  4. our hard-rule check() reports nothing
  5. strictly on canvas: both the plain box and the rotated bounds, with
     zero tolerance
  6. only layout fields changed (x, y, width, height, font_size,
     text_align, corner_radius); nothing removed; content, fonts, colours,
     assets, fit, opacity, rotation, z-order and style runs untouched
  7. images keep their proportions (box aspect within 1% of the source,
     except cover-fit backgrounds, which crop rather than stretch)

Cases: the three samples x the catalogue, random variants of the samples
(eval.stress), and random schema-valid designs (fuzz). Usage:

    python -m eval.validate [N_STRESS_VARIANTS] [N_FUZZ_DESIGNS] [SEED]
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

import jsonschema

from engine import resize as R
from engine.main import self_check
from eval.catalogue import CATALOGUE
from eval.stress import variant

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "chameleon_design_schema_v0.9.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
LAYOUT_FIELDS = {"x", "y", "width", "height", "font_size", "text_align", "corner_radius"}
EPS = 1e-9


def problems(src: dict, tgt: dict) -> list[str]:
    try:
        out = R.resize(copy.deepcopy(src), dict(tgt))
    except Exception as exc:  # resize() must never raise
        return [f"resize raised {exc!r}"]
    errs = []
    if R.resize(copy.deepcopy(src), dict(tgt)) != out:
        errs.append("non-deterministic output")
    errs += [f"schema: {e.message} at {list(e.absolute_path)}" for e in VALIDATOR.iter_errors(out)]
    errs += [f"self_check: {p}" for p in self_check(src, out)]
    errs += [f"check: {p}" for p in R.check(src, out, tgt)["rule"]]
    W, H = tgt["width"], tgt["height"]
    if out.get("canvas") != {"width": W, "height": H}:
        errs.append(f"canvas {out.get('canvas')}")
    by_id = {e["id"]: e for e in src["elements"]}
    for e in out.get("elements", []):
        s = by_id.get(e.get("id"))
        if s is None:
            continue
        for r, kind in ((R.box(e), "box"), (R.rbox(e), "rotated box")):
            if r.x0 < -EPS or r.y0 < -EPS or r.x1 > W + EPS or r.y1 > H + EPS:
                errs.append(f"{e['id']}: {kind} off canvas ({r.x0:.3f},{r.y0:.3f})-({r.x1:.3f},{r.y1:.3f})")
        for k in set(s) | set(e):
            if k in LAYOUT_FIELDS:
                continue
            if k not in e:
                errs.append(f"{e['id']}: field {k} removed")
            elif s.get(k) != e[k]:
                errs.append(f"{e['id']}: field {k} changed {s.get(k)!r} -> {e[k]!r}")
        if e["type"] == "text" and (e["font_size"] != int(e["font_size"]) or e["font_size"] < 1):
            errs.append(f"{e['id']}: font_size {e['font_size']}")
        if e["type"] == "image" and not R.is_background(s, src["canvas"]) and s["width"] > 0 and s["height"] > 0:
            a0 = s["width"] / s["height"]
            a1 = e["width"] / max(e["height"], 1e-9)
            if abs(a1 / a0 - 1) > 0.01:
                errs.append(f"{e['id']}: image aspect {a1:.4f} vs source {a0:.4f}")
    return errs


# ---------------------------------------------------------------- fuzz

FONTS = ["Solstice Serif", "Solstice Sans"]
ASSETS = ["solstice_bg_gradient.png", "solstice_bottle.png", "solstice_logo_lockup.png", "not_in_brand_kit.png"]
WORDS = "summer light golden hour new flavor shop now limited batch crafted sparkling citrus award".split()


def rgba(rng: random.Random) -> str:
    return f"rgba({rng.randrange(256)},{rng.randrange(256)},{rng.randrange(256)},{rng.choice([1, 1, 0.85, 0.5])})"


def fuzz_design(rng: random.Random) -> dict:
    """A random schema-valid design: any mix of texts, images (some
    missing), shapes, groups, rotations, z-orders and optional fields."""
    _, W, H = rng.choice(CATALOGUE)
    els = []
    if rng.random() < 0.7:
        els.append({"id": "bg", "type": "image", "x": 0, "y": 0, "width": W, "height": H,
                    "asset_id": ASSETS[0], "fit": "cover", "z_index": 0})
    for i in range(rng.randint(1, 14)):
        kind = rng.choice(["text", "text", "image", "shape", "unit"])
        w = rng.uniform(0.05, 0.9) * W
        h = rng.uniform(0.03, 0.4) * H
        x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
        base = {"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2),
                "z_index": rng.randint(1, 8)}
        if rng.random() < 0.15:
            base["rotation"] = rng.choice([-15, -8, 5, 12, 90])
        if rng.random() < 0.2:
            base["opacity"] = rng.choice([0.6, 0.9, 1])

        def text(eid, bx):
            words = rng.randint(1, 18)
            t = {"id": eid, "type": "text", **bx, "content": " ".join(rng.choice(WORDS) for _ in range(words)),
                 "font_family": rng.choice(FONTS), "font_size": rng.choice([10, 14, 18, 24, 36, 48, 72, 96]),
                 "color": rgba(rng)}
            if rng.random() < 0.5:
                t["font_weight"] = rng.choice(["regular", "bold"])
            if rng.random() < 0.3:
                t["font_style"] = rng.choice(["normal", "italic"])
            if rng.random() < 0.5:
                t["text_align"] = rng.choice(["left", "center", "right"])
            if rng.random() < 0.3:
                t["line_height"] = rng.choice([1.0, 1.2, 1.5])
            if rng.random() < 0.15:
                end = rng.randint(1, len(t["content"]))
                t["style_runs"] = [{"start": 0, "end": end, "color": rgba(rng)}]
            if rng.random() < 0.1:
                t["max_lines"] = rng.choice([1, 2, None])
            return t

        if kind == "text":
            els.append(text(f"t{i}", base))
        elif kind == "image":
            img = {"id": f"i{i}", "type": "image", **base, "asset_id": rng.choice(ASSETS)}
            if rng.random() < 0.6:
                img["fit"] = rng.choice(["contain", "cover", "fill"])
            els.append(img)
        elif kind == "shape":
            sh = {"id": f"s{i}", "type": "shape", **base, "shape_type": rng.choice(["rectangle", "ellipse", "line"]),
                  "fill_color": rgba(rng)}
            if rng.random() < 0.4:
                sh["corner_radius"] = rng.choice([0, 8, 30])
            if rng.random() < 0.2:
                sh["stroke_color"], sh["stroke_width"] = rgba(rng), 2
            els.append(sh)
        else:  # a button-like unit: pill + label, optionally grouped
            gid = f"g{i}" if rng.random() < 0.6 else None
            pill = {"id": f"u{i}_bg", "type": "shape", **base, "shape_type": "rectangle",
                    "fill_color": rgba(rng), "corner_radius": round(h / 2, 2)}
            label = text(f"u{i}_label", {**base, "x": round(x + 0.1 * w, 2), "y": round(y + 0.2 * h, 2),
                                         "width": round(0.8 * w, 2), "height": round(0.6 * h, 2),
                                         "z_index": base["z_index"] + 1})
            label["content"] = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 3)))
            if gid:
                pill["group_id"] = label["group_id"] = gid
            els += [pill, label]
    return {"canvas": {"width": W, "height": H}, "elements": els}


def main() -> int:
    n_stress = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_fuzz = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    samples = [json.loads(p.read_text())["source"] for p in sorted((ROOT / "input").glob("*.json"))]
    cases = [("sample", s) for s in samples]
    cases += [("stress", variant(rng.choice(samples), rng)) for _ in range(n_stress)]
    fuzz = []
    while len(fuzz) < n_fuzz:
        d = fuzz_design(rng)
        if not list(VALIDATOR.iter_errors(d)):  # only schema-valid inputs
            fuzz.append(("fuzz", d))
    cases += fuzz
    t0 = time.time()
    n = bad = 0
    for kind, src in cases:
        for name, W, H in CATALOGUE:
            n += 1
            errs = problems(src, {"width": W, "height": H})
            if errs:
                bad += 1
                print(f"FAIL {kind} @ {name} {W}x{H}: {errs[:4]}")
                fail_dir = ROOT / "output" / "validate_failures"
                fail_dir.mkdir(parents=True, exist_ok=True)
                (fail_dir / f"fail_{bad:03d}.json").write_text(
                    json.dumps({"source": src, "target_canvas": {"width": W, "height": H}}, indent=1))
    print(f"{n} cases ({len(samples)} samples, {n_stress} stress variants, {n_fuzz} fuzz designs x "
          f"{len(CATALOGUE)} formats) in {time.time() - t0:.0f}s: {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
