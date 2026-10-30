"""Render every input design at every catalogue size into one grid PNG per
design, and print hard-rule results. Usage:

    python -m eval.contact_sheet [--only NAME_SUBSTR] [--candidates]

--candidates also renders each layout archetype's own solution side by side
for the three sample targets, to compare what the scorer chose.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from engine.resize import ARCHETYPES, LAST_FAIL, Ctx, _solve, analyze, build, check, resize, score
from eval.catalogue import CATALOGUE
from renderer.converter.convert import render_design_to_png

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "brand_kit" / "assets"
OUT = ROOT / "output" / "sheets"
CELL = 360  # px, longest side of each thumbnail


def render(design: dict) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        render_design_to_png(design, ASSETS, f.name)
        return Image.open(f.name).convert("RGB")


def sheet(tiles: list[tuple[str, Image.Image]], cols: int = 5) -> Image.Image:
    thumbs = []
    for label, im in tiles:
        s = CELL / max(im.size)
        thumbs.append((label, im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))))
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    pad, lab = 12, 18
    W = cols * (CELL + pad) + pad
    H = sum(max(t.height for _, t in r) + lab + pad for r in rows) + pad
    out = Image.new("RGB", (W, H), (240, 236, 228))
    d = ImageDraw.Draw(out)
    y = pad
    for r in rows:
        x = pad
        for label, t in r:
            d.text((x, y), label, fill=(40, 30, 40))
            out.paste(t, (x, y + lab))
            x += CELL + pad
        y += max(t.height for _, t in r) + lab + pad
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--candidates", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task_path in sorted((ROOT / "input").glob("*.json")):
        if a.only not in task_path.stem:
            continue
        task = json.loads(task_path.read_text())
        src = task["source"]
        tiles = [(f"source {src['canvas']['width']}x{src['canvas']['height']}", render(src))]
        targets = [(n, w, h) for n, w, h in CATALOGUE]
        if a.candidates:
            t = task["target_canvas"]
            targets = [("task target", t["width"], t["height"])]
        for name, w, h in targets:
            target = {"width": w, "height": h}
            if a.candidates:
                st = analyze(src)
                ctx = Ctx(float(w), float(h), st)
                for arch in ARCHETYPES:
                    L = _solve(ARCHETYPES[arch], ctx)
                    if L is None:
                        print(f"  {task_path.stem} {arch}: infeasible ({LAST_FAIL})")
                        continue
                    out = build(src, target, L.out)
                    bad = check(src, out, target)["rule"]
                    tiles.append((f"{arch} t={L.t:.2f} score={score(L):.2f} {'FAIL' if bad else ''}", render(out)))
                    for b in bad:
                        print(f"  {task_path.stem} {arch}: {b}")
                continue
            out = resize(src, target)
            res = check(src, out, target)
            failures += bool(res["rule"])
            tag = "FAIL " if res["rule"] else ""
            tiles.append((f"{tag}{name} {w}x{h}", render(out)))
            for r in res["rule"]:
                print(f"  {task_path.stem} @ {name}: {r}")
        suffix = "_candidates" if a.candidates else ""
        path = OUT / f"{task_path.stem}{suffix}.png"
        sheet(tiles).save(path)
        print(f"wrote {path}")
    print(f"hard-rule failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
