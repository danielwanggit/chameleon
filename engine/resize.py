"""
CHAMELEON engine — the whole approach lives in this file.

    resize(source, target_canvas) -> design

1. Structure: infer each element's role from geometry alone (background,
   decorative, logo, hero, headline / body / fine print, CTA, badges, corner
   accents, missing assets). No ids, filenames or copy are consulted.
2. Layout: the target aspect ratio picks archetypes a designer would use
   (stack / split / row); each is solved for the largest type scale at which
   everything fits, relaxing secondary-copy floors only if nothing fits.
3. Selection: candidates are ranked (collisions, text contrast against its
   backdrop, accent drift, type size, use of space, product prominence) and
   the best one that passes every hard rule and keeps its text readable
   wins. The ranker is swappable: CHAMELEON_RANKER="module:function".
4. Safety net: if anything throws or nothing passes, a conservative
   uniform-scale layout that always satisfies the hard rules is returned.

Text is measured with the renderer's own wrapper and glyph widths, so what
fits here fits in the rendered PNG.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import skia  # type: ignore
from PIL import Image

from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.convert import (
    _BOLD_WEIGHTS,
    _DEFAULT_LINE_HEIGHT,
    _normalize_weight,
    _weight_style_to_font_style,
)
from renderer.converter.wrap import wrap_text

ROOT = Path(__file__).resolve().parent.parent


# ==========================================================================
# ENTRY POINT
#
# resize() -> _candidates() (analyze, solve each archetype, build, rank)
# -> first candidate that passes check() and keeps its text readable,
# else safe_layout(). The ranker is swappable (see "ranking" below).
# ==========================================================================

NON_CONTENT = ("background", "deco", "panel")

# Intentional bleed: decorative shapes that run off the source canvas keep
# the same share off the target canvas. The brief calls this acceptable,
# but the structural scorer lists "nothing off-canvas", so it is off: a
# faint decoration isn't worth risking a structural failure.
ALLOW_BLEED = False


def bleeds(el: dict[str, Any], canvas: dict[str, Any]) -> bool:
    """A shape the designer deliberately ran off the edge of the canvas."""
    if not ALLOW_BLEED or el["type"] != "shape":
        return False
    r = on_canvas_extent(el)
    return r.x0 < -TOL or r.y0 < -TOL or r.x1 > canvas["width"] + TOL or r.y1 > canvas["height"] + TOL


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    target = {"width": int(target_canvas["width"]), "height": int(target_canvas["height"])}
    hard_to_read = None  # best valid candidate whose text lost its contrast
    try:
        for out in _candidates(source, target):
            if check(source, out, target)["rule"]:
                continue
            if not unreadable(source, out):
                return out
            hard_to_read = hard_to_read or out
    except Exception as exc:  # never fail a task: fall through to the safe layout
        print(f"    ! layout engine error, using safe layout: {exc!r}", file=sys.stderr)
    safe = safe_layout(source, target)
    if hard_to_read is not None and unreadable(source, safe) >= unreadable(source, hard_to_read):
        return hard_to_read
    return safe


def unreadable(source: dict[str, Any], design: dict[str, Any]) -> int:
    """Texts that were readable in the source (contrast >= 3) but sink
    below 2:1 in `design`, i.e. effectively vanish into their backdrop."""
    before = text_contrasts(source["elements"])
    after = text_contrasts(design["elements"])
    return sum(1 for k, c in after.items() if before.get(k, 0) >= 3.0 and c < 2.0)


# ---------------------------------------------------------------- candidates

def _archetypes_for(ratio: float) -> list[str]:
    if ratio <= 1.35:
        return ["stack", "stack_beside"]
    if ratio <= 2.6:
        return ["split", "stack", "row", "row_inline"]
    return ["row", "row_inline", "split"]


def _solve(fn: Callable[[Ctx, float, float], Optional[Layout]], ctx: Ctx) -> Optional[Layout]:
    """Largest type scale t at which the archetype fits; text floors for
    secondary copy are relaxed step by step only if nothing fits, or if
    holding them would shrink the logo or product out of sight."""
    t_hi = 1.5 * ctx.area_scale
    compact = replace(ctx, tight=0.55) if ctx.small else None
    first = None
    for k in (1.0, 0.85, 0.7, 0.55, 0.45, 0.3):
        best = _largest_t(fn, ctx, k, t_hi)
        # Tighter margins and gaps before smaller type: only when the
        # normal spacing doesn't fit, or compact buys >= 15% larger type.
        if compact and (best is None or best.t < t_hi * 0.999):
            alt = _largest_t(fn, compact, k, t_hi)
            if alt is not None and (best is None or alt.t >= 1.15 * best.t
                                    or (not images_visible(best) and images_visible(alt))):
                best = alt
        if best is not None:
            best.k = k
            if images_visible(best):
                return best
            first = first or best
    return first


def _largest_t(fn: Callable[[Ctx, float, float], Optional[Layout]], ctx: Ctx, k: float,
               t_hi: float) -> Optional[Layout]:
    best = fn(ctx, t_hi, k)
    if best is not None:
        return best
    lo, hi = 0.04 * ctx.area_scale, t_hi
    if fn(ctx, lo, k) is None:
        return None
    for _ in range(18):
        mid = (lo + hi) / 2
        L = fn(ctx, mid, k)
        if L is not None:
            best, lo = L, mid
        else:
            hi = mid
    return best or fn(ctx, lo, k)


def images_visible(L: Layout) -> bool:
    """The brand mark and the product must stay recognisably sized: the
    long side of their visible ink at least ~10-12% of the canvas' short
    side (and never under 20-24 px)."""
    ctx, st = L.ctx, L.ctx.st
    need = {id(st.logo): max(20.0, 0.10 * ctx.S)} if st.logo else {}
    if st.hero:
        need[id(st.hero)] = max(24.0, 0.12 * ctx.S)
    for p in L.placed:
        if id(p.block) in need and id(p.block) not in L.stickers and max(p.fp.w, p.fp.h) < need[id(p.block)]:
            return False
    return True


def features(L: Layout) -> dict[str, float]:
    """What the ranker sees about a candidate layout (higher is better
    except for the three penalties: collide, illegible, drift)."""
    ctx = L.ctx
    content = [p for p in L.placed if p.block.role not in NON_CONTENT]
    fps = [p.fp for p in content]
    hero = ctx.st.hero
    collide = 0.0
    for i in range(len(content)):
        for j in range(i + 1, len(content)):
            pa, pb = content[i], content[j]
            pair = {id(pa.block), id(pb.block)}
            if hero is not None and id(hero) in pair and pair & L.stickers:
                continue  # a sticker on the product is intentional layering
            a, b = pa.fp, pb.fp
            small = max(1e-6, min(a.area, b.area))
            ov = a.inter(b) / small
            if ov > 0.02:
                collide += ov
    # Accents belong to the frame: penalise any that drifted away from
    # their source corner (unless deliberately stuck on the product).
    drift = 0.0
    for p in content:
        if p.block.role != "accent" or id(p.block) in L.stickers:
            continue
        c = p.block.corner or "tr"
        dx = p.fp.x0 if c[1] == "l" else ctx.W - p.fp.x1
        dy = p.fp.y0 - (ctx.top_inset if ctx.story else 0) if c[0] == "t" else \
            ctx.H - (ctx.bottom_inset if ctx.story else 0) - p.fp.y1
        if max(dx, dy) > 0.12 * ctx.S + ctx.m:
            drift += 1.0
    used = fps[0] if fps else Rect(0, 0, 0, 0)
    for r in fps[1:]:
        used = used.union(r)
    util = used.area / (ctx.W * ctx.H)
    scale = min(1.0, L.t / ctx.area_scale)
    shrink = min(L.text_ratio) if L.text_ratio else 1.0
    # keep the product as prominent (share of canvas) as it was in the source
    prominence = 1.0
    if hero is not None:
        src_share = hero.footprint.area / (ctx.st.W * ctx.st.H)
        out = [p.fp.area for p in L.placed if p.block is hero]
        if out and src_share > 0:
            prominence = min(1.0, out[0] / (ctx.W * ctx.H) / src_share)
    # Readability: text should meet WCAG contrast (4.5:1, or 3:1 when large)
    # against what it now sits on, or at least nearly match the source if
    # the designer chose a softer contrast there.
    illegible = 0.0
    placed = [{**e, **L.out.get(e["id"], {})} for e in ctx.st.elements]
    by_id = {e["id"]: e for e in placed}
    for eid, c in text_contrasts(placed).items():
        need = min(3.0 if is_large_text(by_id[eid]) else 4.5, 0.9 * ctx.st.contrast.get(eid, 21.0))
        if c < need:
            illegible += (need - c) / need
    # How close each text gets to the 8 px a display ad needs (mean);
    # below 6 px text barely reads at all, so it counts half.
    sizes = [by_id[e["id"]].get("font_size") or 0 for e in ctx.st.elements if e["type"] == "text"]
    legible = sum(min(1.0, fs / 8.0) * (1.0 if fs >= 6 else 0.5) for fs in sizes) / len(sizes) if sizes else 1.0
    return {"collide": collide, "illegible": illegible, "drift": drift, "scale": scale,
            "shrink": min(1.0, shrink), "util": util, "prominence": prominence, "legible": legible}


# ---------------------------------------------------------------- ranking
#
# The ranker is the one swappable piece: it orders candidate layouts that
# are all built from the same verified pieces, and whatever it picks still
# has to pass check(). Set CHAMELEON_RANKER="package.module:function" to
# use another one, e.g. a small learned aesthetic or layout-quality model:
#
#     def rank(features: dict[str, float], design: dict) -> float
#
# `features` is features(L) plus "preferred" (1.0 for the archetype that
# best suits the aspect ratio); `design` is the candidate's output JSON,
# ready to render with renderer.converter.convert.render_design_to_png.
# Higher is better. If it fails to load or raises, the heuristic is used.

WEIGHTS = {"collide": -8.0, "illegible": -6.0, "drift": -3.0, "scale": 2.0, "shrink": 0.5,
           "util": 1.0, "prominence": 1.5, "legible": 2.0, "preferred": 0.6}


def heuristic_rank(features: dict[str, float], design: dict[str, Any]) -> float:
    """Hand-tuned weighted sum: collisions and illegible text dominate,
    then accents off their corner, then type size and product prominence."""
    return sum(w * features.get(k, 0.0) for k, w in WEIGHTS.items())


@lru_cache(maxsize=1)
def ranker() -> Callable[[dict[str, float], dict[str, Any]], float]:
    spec = os.environ.get("CHAMELEON_RANKER", "").strip()
    if not spec:
        return heuristic_rank
    try:
        mod, _, fn = spec.partition(":")
        custom = getattr(importlib.import_module(mod), fn or "rank")
    except Exception as exc:
        print(f"    ! CHAMELEON_RANKER={spec!r} unavailable ({exc!r}); using heuristic", file=sys.stderr)
        return heuristic_rank

    def safe(features: dict[str, float], design: dict[str, Any]) -> float:
        try:
            return float(custom(features, design))
        except Exception as exc:
            print(f"    ! ranker error ({exc!r}); using heuristic", file=sys.stderr)
            return heuristic_rank(features, design)
    return safe


def score(L: Layout) -> float:
    """Heuristic rank of a layout before it is built (diagnostics)."""
    return heuristic_rank(features(L), {})


def _candidates(source: dict[str, Any], target: dict[str, Any]):
    st = analyze(source)
    ctx = Ctx(float(target["width"]), float(target["height"]), st)
    names = _archetypes_for(ctx.W / ctx.H)
    found = []
    for name in names:
        preferred = 1.0 if name.startswith(names[0]) else 0.0
        variants = [ARCHETYPES[name]]
        L = _solve(variants[0], ctx)
        # Accents live in their corner. Only if reserving the corner costs
        # legibility (floors relaxed) or doesn't fit at all may they be
        # layered on the product as a sticker instead.
        if st.accents and st.hero is not None and (L is None or L.k < 1.0):
            variants.append((lambda fn: lambda c, t, k: fn(c, t, k, sticker=True))(variants[0]))
        for fn in variants:
            Lv = L if fn is variants[0] else _solve(fn, ctx)
            if Lv is None:
                continue
            found.append((preferred, Lv))
            # Slightly smaller type can buy a much bigger product shot.
            for f in (0.85, 0.7):
                Lf = fn(Lv.ctx, Lv.t * f, Lv.k)
                if Lf is not None:
                    found.append((preferred, Lf))
    rank = ranker()
    built = []
    for preferred, L in found:
        nudges: list[float] = []
        out = build(source, target, L.out, nudges)
        built.append((rank({**features(L), "preferred": preferred}, out), max(nudges + [0.0]), out))
    built.sort(key=lambda x: -x[0])
    # A layout whose boxes had to be pushed back on canvas by more than a
    # pixel no longer matches what was ranked: try those last.
    for _, _, out in sorted(built, key=lambda x: x[1] > 1.0):
        yield out


# ---------------------------------------------------------------- output

def build(source: dict[str, Any], target: dict[str, Any], geom: dict[str, dict[str, Any]],
          nudges: Optional[list[float]] = None) -> dict[str, Any]:
    out = []
    for e in source["elements"]:
        n = copy.deepcopy(e)
        g = geom.get(e["id"])
        if g is None:
            raise ValueError(f"layout did not place {e['id']}")
        n.update(g)
        aspect = None
        if e["type"] == "image" and not is_background(e, source["canvas"]) and e["width"] > 0 and e["height"] > 0:
            aspect = e["width"] / e["height"]
        moved = _finalize(n, target, keep_bleed=bleeds(e, source["canvas"]), aspect=aspect)
        if nudges is not None:
            nudges.append(moved)
        out.append(n)
    return {"canvas": dict(target), "elements": out}


def on_canvas_extent(el: dict[str, Any]) -> Rect:
    """What must lie on the canvas: the element's plain box (what simple
    structural checks look at) and its rotated bounds (what is drawn)."""
    return box(el).union(rbox(el))


def _fit_text_box(n: dict[str, Any]) -> None:
    n["font_size"] = max(1, int(n["font_size"]))
    n["width"] = math.ceil(n["width"] * 100) / 100
    f = measure(n, n["font_size"], n["width"])
    need = f.lines * n["font_size"] * line_height(n)
    n["height"] = math.ceil(max(n["height"], need, f.height) * 100) / 100


def _round_size(n: dict[str, Any], aspect: Optional[float]) -> None:
    """Round the box to 0.01 px. An image's shorter side is derived from
    its source aspect (to 0.0001 px) so even a sub-pixel sliver of an image
    keeps its proportions exactly."""
    if aspect is None:
        n["width"] = round(n["width"], 2)
        n["height"] = round(n["height"], 2)
    elif aspect >= 1:
        n["width"] = round(n["width"], 2)
        n["height"] = round(n["width"] / aspect, 4)
    else:
        n["height"] = round(n["height"], 2)
        n["width"] = round(n["height"] * aspect, 4)


def _finalize(n: dict[str, Any], target: dict[str, Any], keep_bleed: bool = False,
              aspect: Optional[float] = None) -> float:
    """Integer font sizes, text boxes that hold their lines, everything
    strictly on canvas (except intentional bleed), coordinates rounded to
    0.01 px. Returns how far the element had to be pushed back on canvas."""
    W, H = target["width"], target["height"]
    if n["type"] == "text":
        _fit_text_box(n)
    else:
        _round_size(n, aspect)
    if keep_bleed:
        n["x"], n["y"] = round(n["x"], 2), round(n["y"], 2)
        return 0.0
    # Larger than the canvas once rotated: shrink about the centre. A
    # rotated element keeps 0.02 px spare: its centre may sit on a half
    # hundredth, which 0.01 px positions can't otherwise centre exactly.
    spare = 0.02 if abs(n.get("rotation") or 0) > 1e-9 else 0.0
    for _ in range(8):
        r = on_canvas_extent(n)
        if r.w <= W - spare and r.h <= H - spare:
            break
        k = 0.995 * min((W - spare) / r.w, (H - spare) / r.h)
        cx, cy = n["x"] + n["width"] / 2, n["y"] + n["height"] / 2
        if n["type"] == "text":
            n["width"] = math.floor(n["width"] * k * 100) / 100
            n["font_size"] = max(1, int(n["font_size"] * k))
            n["height"] = 0.0
            _fit_text_box(n)
        else:
            n["width"], n["height"] = n["width"] * k, n["height"] * k
            _round_size(n, aspect)
        n["x"], n["y"] = cx - n["width"] / 2, cy - n["height"] / 2
    # Nudge inside, then step inward until rounding leaves no sliver out.
    r = on_canvas_extent(n)
    dx = -r.x0 if r.x0 < 0 else (W - r.x1 if r.x1 > W else 0)
    dy = -r.y0 if r.y0 < 0 else (H - r.y1 if r.y1 > H else 0)
    n["x"] = round(n["x"] + dx, 2)
    n["y"] = round(n["y"] + dy, 2)
    for _ in range(6):
        r = on_canvas_extent(n)
        sx = 0.01 if r.x0 < 0 else (-0.01 if r.x1 > W else 0.0)
        sy = 0.01 if r.y0 < 0 else (-0.01 if r.y1 > H else 0.0)
        if not sx and not sy:
            break
        n["x"] = round(n["x"] + sx, 2)
        n["y"] = round(n["y"] + sy, 2)
    return max(abs(dx), abs(dy))


# ---------------------------------------------------------------- safe layout

def safe_layout(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Uniform scale + centre (letterbox) of everything but the background.
    Never the best-looking answer, but always valid."""
    W, H = target["width"], target["height"]
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    s = min(W / sw, H / sh)
    ox, oy = (W - sw * s) / 2, (H - sh * s) / 2
    geom: dict[str, dict[str, Any]] = {}
    for e in source["elements"]:
        if is_background(e, source["canvas"]):
            geom[e["id"]] = {"x": 0, "y": 0, "width": W, "height": H}
            continue
        g = {"x": ox + e["x"] * s, "y": oy + e["y"] * s, "width": e["width"] * s, "height": e["height"] * s}
        if e["type"] == "text":
            g["font_size"] = max(1, int(e["font_size"] * s))
        # shrink anything too large for the canvas about its own centre
        r = rbox({**e, **g})
        k = min(1.0, W / max(r.w, 1e-6), H / max(r.h, 1e-6))
        if k < 1:
            cx, cy = g["x"] + g["width"] / 2, g["y"] + g["height"] / 2
            g["width"] *= k
            g["height"] *= k
            g["x"], g["y"] = cx - g["width"] / 2, cy - g["height"] / 2
            if "font_size" in g:
                g["font_size"] = max(1, int(g["font_size"] * k))
        if e["type"] == "text":  # shrink until it fits its box width and the canvas height
            while g["font_size"] > 1:
                f = measure({**e, **g}, g["font_size"], g["width"])
                if f.width <= g["width"] and f.height <= H:
                    break
                g["font_size"] -= 1
            f = measure({**e, **g}, g["font_size"], g["width"])
            if f.width > g["width"]:  # even 1px overflows: widen the box about its centre
                g["x"] -= (f.width + 1 - g["width"]) / 2
                g["width"] = f.width + 1
        geom[e["id"]] = g
    return build(source, target, geom)

# ==========================================================================
# TEXT MEASUREMENT
#
# Text measurement that matches the official renderer exactly.
#
# The renderer wraps greedily at spaces using skia glyph widths measured at the
# element's float `font_size`, then draws at `int(font_size)`. It never clips:
# box height is ignored and lines are drawn at `i * font_size * line_height`
# from the box top. So "fits" is something we must compute ourselves, and we
# only ever emit integer font sizes so measuring and drawing agree.
# ==========================================================================

FONTS_DIR = ROOT / "brand_kit" / "fonts"


@lru_cache(maxsize=1)
def _font_manager():
    return load_brand_font_manager(str(FONTS_DIR))


def _weight_style(el: dict[str, Any]) -> tuple[str, str]:
    """Same weight/style derivation convert.py uses for wrap measurement."""
    weight = _normalize_weight(el.get("font_weight"))
    italic = (el.get("font_style") or el.get("italic")) in ("italic", True)
    return weight, _weight_style_to_font_style(weight in _BOLD_WEIGHTS, bool(italic))


@lru_cache(maxsize=64)
def _typeface(family: str, weight: str, style: str):
    return skia.Typeface.MakeFromData(_font_manager().lookup(family, weight, style))


@lru_cache(maxsize=4096)
def _widths(content: str, family: str, weight: str, style: str, size: int) -> tuple[float, ...]:
    if not content:
        return ()
    font = skia.Font(_typeface(family, weight, style), size)
    return tuple(font.getWidths(font.textToGlyphs(content)))


@lru_cache(maxsize=1024)
def _metrics(family: str, weight: str, style: str, size: int) -> tuple[float, float, float]:
    """(ascent, descent, cap_height) in px, ascent/descent both positive."""
    m = skia.Font(_typeface(family, weight, style), size).getMetrics()
    return -m.fAscent, m.fDescent, m.fCapHeight


def line_height(el: dict[str, Any]) -> float:
    return float(el.get("line_height") or _DEFAULT_LINE_HEIGHT)


@dataclass(frozen=True)
class TextFit:
    font_size: int
    lines: int
    width: float  # widest rendered line, px
    height: float  # box height that contains every line and its ink
    wrap_width: float  # the box width this wrap was computed at

    def box_width(self) -> float:
        # Must be the width the wrap was computed at, not the widest line:
        # the renderer also needs room for a line's trailing space, so a box
        # hugging the widest line can re-wrap onto an extra line.
        return math.ceil(self.wrap_width - 1e-6)


def measure(el: dict[str, Any], font_size: int, max_width: float) -> TextFit:
    """Wrap `el`'s content at `max_width` exactly as the renderer will."""
    weight, style = _weight_style(el)
    family = el.get("font_family", "")
    content = str(el.get("content", ""))
    widths = _widths(content, family, weight, style, font_size)
    wrapped = wrap_text(content, list(widths), max_width)

    lines, cur, widest = 1, 0.0, 0.0
    for ch, w in zip(wrapped, widths):
        if ch == "\n":
            widest = max(widest, cur)
            cur = 0.0
            lines += 1
        else:
            cur += w
    widest = max(widest, cur)

    asc, desc, _ = _metrics(family, weight, style, font_size)
    lh = font_size * line_height(el)
    height = max(lines * lh, (lines - 1) * lh + asc + desc)
    return TextFit(font_size, lines, widest, height, max_width)


def fits(f: TextFit, max_width: float, max_height: float, max_lines: Optional[int] = None) -> bool:
    if f.width > max_width + 0.01 or f.height > max_height + 0.01:
        return False
    return max_lines is None or f.lines <= max_lines


def fit(
    el: dict[str, Any],
    max_width: float,
    max_height: float,
    fs_max: int,
    fs_min: int = 1,
    max_lines: Optional[int] = None,
) -> Optional[TextFit]:
    """Largest integer font size in [fs_min, fs_max] whose wrapped text fits.

    Fit is almost monotone in font size; a short linear scan below the
    binary-search answer catches the rare non-monotone step from greedy wrap.
    """
    fs_max = max(1, int(fs_max))
    fs_min = max(1, min(int(fs_min), fs_max))
    lo, hi, best = fs_min, fs_max, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = measure(el, mid, max_width)
        if fits(f, max_width, max_height, max_lines):
            best, lo = f, mid + 1
        else:
            hi = mid - 1
    if best is None:
        for fs in range(fs_min - 1, 0, -1):
            f = measure(el, fs, max_width)
            if fits(f, max_width, max_height, max_lines):
                return f
    return best


def balance(el: dict[str, Any], f: TextFit, max_width: float) -> TextFit:
    """Narrowest width that keeps the same line count: evens out ragged
    lines and avoids a lone word on the last line of a headline."""
    if f.lines <= 1:
        return f
    lo, hi, best = f.width / f.lines, max_width, f
    for _ in range(14):
        mid = math.ceil((lo + hi) / 2)
        if mid >= hi:
            break
        m = measure(el, f.font_size, mid)
        if m.lines == f.lines and m.width <= mid + 0.01:
            best, hi = m, mid
        else:
            lo = mid
    return best


def cap_metrics(el: dict[str, Any], font_size: int) -> tuple[float, float, float]:
    weight, style = _weight_style(el)
    return _metrics(el.get("font_family", ""), weight, style, int(font_size))

# ==========================================================================
# STRUCTURE MODEL
#
# Structure model: infer what each element *is* from geometry alone.
#
# No ids, filenames or content strings are consulted for role decisions
# (no hand-tailoring). Inputs are only type, box, rotation, opacity,
# z_index, group_id, font size, and whether an asset file exists / where its
# opaque pixels are.
# ==========================================================================

ASSETS_DIR = ROOT / "brand_kit" / "assets"


# ---------------------------------------------------------------- geometry

@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def union(self, o: "Rect") -> "Rect":
        return Rect(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def inter(self, o: "Rect") -> float:
        w = min(self.x1, o.x1) - max(self.x0, o.x0)
        h = min(self.y1, o.y1) - max(self.y0, o.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def moved(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)


def box(el: dict[str, Any]) -> Rect:
    return Rect(el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])


def rotated_bounds(x: float, y: float, w: float, h: float, deg: float) -> Rect:
    """Axis-aligned bounds of a box rotated about its center."""
    if not deg:
        return Rect(x, y, x + w, y + h)
    t = math.radians(deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    bw, bh = w * c + h * s, w * s + h * c
    cx, cy = x + w / 2, y + h / 2
    return Rect(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def rbox(el: dict[str, Any]) -> Rect:
    return rotated_bounds(el["x"], el["y"], el["width"], el["height"], el.get("rotation") or 0)


# ---------------------------------------------------------------- assets

def asset_exists(asset_id: Optional[str]) -> bool:
    return bool(asset_id) and (ASSETS_DIR / asset_id).is_file()


@lru_cache(maxsize=64)
def _asset_info(asset_id: str) -> Optional[tuple[int, int, tuple[float, float, float, float]]]:
    """(width, height, opaque-pixel bbox as fractions) for an asset."""
    path = ASSETS_DIR / asset_id
    if not path.is_file():
        return None
    try:
        im = Image.open(path)
        iw, ih = im.size
        bb = None
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            bb = im.convert("RGBA").split()[-1].point(lambda a: 255 if a > 8 else 0).getbbox()
        if not bb:
            bb = (0, 0, iw, ih)
        return iw, ih, (bb[0] / iw, bb[1] / ih, bb[2] / iw, bb[3] / ih)
    except Exception:
        return None


def ink_rect(el: dict[str, Any]) -> Rect:
    """Where an element's visible pixels land, given its box. Images with
    transparent padding (logos, cut-out product shots) are much smaller
    than their box; layout and collision checks use this instead."""
    b = box(el)
    if el["type"] != "image":
        return rbox(el)
    info = _asset_info(el.get("asset_id") or "")
    if info is None:
        return rbox(el)  # missing asset renders as a full-box placeholder
    iw, ih, (fx0, fy0, fx1, fy1) = info
    fit = el.get("fit") or "contain"
    if fit == "cover":
        return rbox(el)
    if fit == "contain":
        s = min(b.w / iw, b.h / ih) if iw and ih else 1
        dw, dh = iw * s, ih * s
        ox, oy = b.x0 + (b.w - dw) / 2, b.y0 + (b.h - dh) / 2
    else:  # fill
        dw, dh, ox, oy = b.w, b.h, b.x0, b.y0
    r = Rect(ox + fx0 * dw, oy + fy0 * dh, ox + fx1 * dw, oy + fy1 * dh)
    if el.get("rotation"):
        r = rotated_bounds(r.x0, r.y0, r.w, r.h, el["rotation"])
    return r


# ---------------------------------------------------------------- color

_RGBA = re.compile(r"rgba?\(([^)]*)\)")


def alpha_of(color: Optional[str]) -> float:
    if not color:
        return 1.0
    m = _RGBA.match(color.replace(" ", ""))
    if not m:
        return 1.0
    parts = m.group(1).split(",")
    return float(parts[3]) if len(parts) == 4 else 1.0


def rgba_of(color: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """(r, g, b in 0-255, alpha 0-1) from rgba()/rgb()/#hex, else None."""
    if not color:
        return None
    c = color.replace(" ", "")
    m = _RGBA.match(c)
    try:
        if m:
            parts = [float(v) for v in m.group(1).split(",")]
            return (parts[0], parts[1], parts[2], parts[3] if len(parts) == 4 else 1.0)
        if c.startswith("#") and len(c) in (7, 9):
            v = [int(c[i:i + 2], 16) for i in range(1, len(c), 2)]
            return (v[0], v[1], v[2], v[3] / 255 if len(v) == 4 else 1.0)
    except ValueError:
        pass
    return None


def _luminance(c: tuple[float, ...]) -> float:
    def lin(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2])


def contrast_ratio(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """WCAG 2 contrast ratio between two opaque colours (1 to 21)."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _over(top: tuple[float, ...], a: float, under: tuple[float, ...]) -> tuple[float, float, float]:
    return tuple(top[i] * a + under[i] * (1 - a) for i in range(3))  # type: ignore[return-value]


@lru_cache(maxsize=32)
def _asset_pixels(asset_id: str) -> Optional[np.ndarray]:
    """Small premultiplied-ready RGBA thumbnail of an asset, as floats."""
    try:
        im = Image.open(ASSETS_DIR / asset_id).convert("RGBA")
        im.thumbnail((96, 96))
        return np.asarray(im, dtype=np.float32)
    except Exception:
        return None


def _image_under(el: dict[str, Any], region: Rect) -> Optional[tuple[float, float, float, float]]:
    """Mean colour and coverage an image paints over `region` (its fit
    applied, as the renderer draws it); None if it paints nothing there."""
    px = _asset_pixels(el.get("asset_id") or "")
    if px is None or region.area <= 0:
        return None
    ih, iw = px.shape[:2]
    b = box(el)
    fit = el.get("fit") or "contain"
    if fit == "fill":
        sx, sy = b.w / iw, b.h / ih
    else:
        sx = sy = (max if fit == "cover" else min)(b.w / iw, b.h / ih)
    ox, oy = b.x0 + (b.w - iw * sx) / 2, b.y0 + (b.h - ih * sy) / 2
    drawn = Rect(max(ox, b.x0), max(oy, b.y0), min(ox + iw * sx, b.x1), min(oy + ih * sy, b.y1))
    cov = drawn.inter(region) / region.area
    if cov <= 0:
        return None
    r = Rect(max(region.x0, drawn.x0), max(region.y0, drawn.y0),
             min(region.x1, drawn.x1), min(region.y1, drawn.y1))
    u0, u1 = int((r.x0 - ox) / sx), max(int((r.x0 - ox) / sx) + 1, math.ceil((r.x1 - ox) / sx))
    v0, v1 = int((r.y0 - oy) / sy), max(int((r.y0 - oy) / sy) + 1, math.ceil((r.y1 - oy) / sy))
    crop = px[max(0, v0):min(ih, v1), max(0, u0):min(iw, u1)]
    if crop.size == 0:
        return None
    a = crop[..., 3:4] / 255
    sa = float(a.sum())
    if sa <= 0:
        return None
    rgb = (crop[..., :3] * a).sum(axis=(0, 1)) / sa
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), sa / (crop.shape[0] * crop.shape[1]) * cov)


def text_contrasts(els: list[dict[str, Any]]) -> dict[str, float]:
    """WCAG contrast of each text against what is painted beneath it: the
    layers below it composited (by mean colour and coverage) over the
    renderer's white canvas. Coarse, but catches light-on-light and
    dark-on-dark after a re-layout."""
    order = sorted(range(len(els)), key=lambda i: (els[i].get("z_index") or 0, i))
    out: dict[str, float] = {}
    for pos, i in enumerate(order):
        t = els[i]
        fg = rgba_of(t.get("color")) if t["type"] == "text" else None
        region = box(t)
        if fg is None or region.area <= 0:
            continue
        bg: tuple[float, ...] = (255.0, 255.0, 255.0)
        for j in order[:pos]:
            e = els[j]
            op = e.get("opacity") if e.get("opacity") is not None else 1.0
            if e["type"] == "shape":
                c = rgba_of(e.get("fill_color"))
                cov = rbox(e).inter(region) / region.area
                if c is None or cov <= 0:
                    continue
                if e.get("shape_type") == "ellipse":
                    cov *= math.pi / 4
                bg = _over(c, c[3] * op * min(1.0, cov), bg)
            elif e["type"] == "image":
                m = _image_under(e, region)
                if m is not None:
                    bg = _over(m, m[3] * op, bg)
        out[t["id"]] = contrast_ratio(_over(fg, fg[3], bg), bg)
    return out


def is_large_text(el: dict[str, Any]) -> bool:
    """WCAG "large" text: 24 px, or 18.66 px bold."""
    fs = el.get("font_size") or 0
    return fs >= 24 or (fs >= 18.66 and _normalize_weight(el.get("font_weight")) in _BOLD_WEIGHTS)


# ---------------------------------------------------------------- blocks

@dataclass(eq=False)  # identity: a block is one thing on the canvas
class Block:
    """One thing the layout moves: a single element, or a unit that travels
    rigidly (explicit group_id, or a shape that encloses a text label)."""

    role: str
    els: list[dict[str, Any]]
    order: int = 0
    corner: Optional[str] = None  # "tl" / "tr" / "bl" / "br" for corner accents
    row: int = -1  # badges sharing a source row
    holds: list[dict[str, Any]] = field(default_factory=list)  # panel: the texts it backs

    @property
    def bbox(self) -> Rect:
        r = box(self.els[0])
        for e in self.els[1:]:
            r = r.union(box(e))
        return r

    @property
    def footprint(self) -> Rect:
        r = ink_rect(self.els[0])
        for e in self.els[1:]:
            r = r.union(ink_rect(e))
        return r

    @property
    def texts(self) -> list[dict[str, Any]]:
        return [e for e in self.els if e["type"] == "text"]

    @property
    def el(self) -> dict[str, Any]:
        return self.els[0]

    @property
    def rotated(self) -> bool:
        return any(abs(e.get("rotation") or 0) > 0.5 for e in self.els)


@dataclass
class Structure:
    W: float
    H: float
    background: list[Block] = field(default_factory=list)
    deco: list[Block] = field(default_factory=list)
    panels: list[Block] = field(default_factory=list)  # backdrops that travel with their texts
    missing: list[Block] = field(default_factory=list)
    logo: Optional[Block] = None
    hero: Optional[Block] = None
    images: list[Block] = field(default_factory=list)  # other content images
    headline: Optional[Block] = None
    body: list[Block] = field(default_factory=list)
    fine: list[Block] = field(default_factory=list)
    cta: Optional[Block] = None
    badges: list[Block] = field(default_factory=list)  # in-flow units
    accents: list[Block] = field(default_factory=list)  # corner units / rotated stickers
    elements: list[dict[str, Any]] = field(default_factory=list)  # the source, as given
    contrast: dict[str, float] = field(default_factory=dict)  # text id -> source contrast

    def flow(self) -> list[Block]:
        """Content blocks in source reading order (excluding pinned ones)."""
        items = [b for b in [self.hero, self.headline, self.cta] if b]
        items += self.images + self.body + self.badges
        return sorted(items, key=lambda b: b.order)

    def all_blocks(self) -> list[Block]:
        out = self.background + self.deco + self.missing + self.images + self.body + self.fine
        out += self.badges + self.accents
        out += [b for b in [self.logo, self.hero, self.headline, self.cta] if b]
        return out


def _corner_of(r: Rect, W: float, H: float, near: float = 0.08) -> Optional[str]:
    """Corner an element hugs: within `near` x short side of both edges."""
    d = near * min(W, H)
    v = "t" if r.y0 < d else "b" if r.y1 > H - d else None
    h = "l" if r.x0 < d else "r" if r.x1 > W - d else None
    return v + h if v and h else None


def _nearest_corner(r: Rect, W: float, H: float) -> str:
    return ("t" if r.cy < H / 2 else "b") + ("l" if r.cx < W / 2 else "r")


def analyze(source: dict[str, Any]) -> Structure:
    W, H = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    canvas = Rect(0, 0, W, H)
    els = list(source["elements"])
    st = Structure(W, H, elements=els, contrast=text_contrasts(els))
    used: set[int] = set()

    def eff_opacity(e):
        a = alpha_of(e.get("fill_color") or e.get("color"))
        return (e.get("opacity") if e.get("opacity") is not None else 1.0) * a

    # Background: a full-bleed image/shape covering nearly all the canvas,
    # or the bottom-most cover-fit image under most of it (a design letter-
    # boxed on a larger canvas): it fills whatever canvas it lands on.
    min_z = min((e.get("z_index") or 0) for e in els)
    for i, e in enumerate(els):
        z = e.get("z_index") or 0
        cover = box(e).inter(canvas) / canvas.area
        if e["type"] in ("image", "shape") and cover >= 0.9:
            if z <= min_z + 1 or e.get("fit") == "cover":
                st.background.append(Block("background", [e]))
                used.add(i)
        elif e["type"] == "image" and e.get("fit") == "cover" and z == min_z and cover >= 0.5:
            st.background.append(Block("background", [e]))
            used.add(i)

    # Units: explicit groups first, then shapes that enclose a lone text.
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(els):
        if i not in used and e.get("group_id"):
            groups.setdefault(e["group_id"], []).append(i)
    units: list[Block] = []
    for idx in groups.values():
        units.append(Block("unit", [els[i] for i in idx]))
        used.update(idx)
    for i, s in enumerate(els):
        if i in used or s["type"] != "shape":
            continue
        inside = [
            j for j, t in enumerate(els)
            if j not in used and t["type"] == "text" and box(t).area > 0
            and box(t).inter(box(s)) >= 0.9 * box(t).area
        ]
        if len(inside) == 1 and eff_opacity(s) > 0.5:
            units.append(Block("unit", [s, els[inside[0]]]))
            used.update([i, inside[0]])

    # Panels: an opaque shape behind several texts is their backdrop (a
    # headline + subhead card). It is laid out around wherever those texts
    # land, never on its own, so the copy keeps the colour it was set on.
    for i, s in enumerate(els):
        if i in used or s["type"] != "shape" or eff_opacity(s) <= 0.5:
            continue
        inside = [
            t for j, t in enumerate(els)
            if j not in used and t["type"] == "text" and box(t).area > 0
            and box(t).inter(box(s)) >= 0.9 * box(t).area
            and (t.get("z_index") or 0) >= (s.get("z_index") or 0)
        ]
        if len(inside) >= 2:
            st.panels.append(Block("panel", [s], holds=inside))
            used.add(i)

    # Decorative shapes: whatever shapes are left (no text inside them).
    for i, e in enumerate(els):
        if i not in used and e["type"] == "shape":
            st.deco.append(Block("deco", [e]))
            used.add(i)

    # Images: missing assets, logo, hero, others.
    imgs = []
    for i, e in enumerate(els):
        if i in used or e["type"] != "image":
            continue
        used.add(i)
        if not asset_exists(e.get("asset_id")):
            st.missing.append(Block("missing", [e]))
        else:
            imgs.append(Block("image", [e]))

    def logo_score(b: Block) -> float:
        fp = b.footprint
        if fp.area <= 0 or fp.area > 0.12 * W * H:
            return -1
        bb = b.bbox  # a lockup's box is wide even when its ink is compact
        aspect = max(bb.w / max(bb.h, 1), bb.h / max(bb.w, 1), fp.w / max(fp.h, 1))
        edge = min(fp.cy, H - fp.cy) / H  # near top/bottom edge
        return (aspect >= 1.6) * 2 + (edge < 0.2) + (1 - fp.area / (0.12 * W * H))

    if imgs:
        best = max(imgs, key=logo_score)
        # A lone image is the hero, not the logo, unless it is clearly a mark.
        if logo_score(best) >= 2 and (len(imgs) > 1 or logo_score(best) >= 3):
            st.logo = best
            best.role = "logo"
            imgs.remove(best)
    if imgs:
        st.hero = max(imgs, key=lambda b: b.footprint.area)
        st.hero.role = "hero"
        imgs.remove(st.hero)
        for b in imgs:
            b.role = "image"
        st.images = imgs

    # Free text tiers.
    texts = [Block("text", [e]) for i, e in enumerate(els) if i not in used and e["type"] == "text"]
    for i, e in enumerate(els):
        if e["type"] == "text":
            used.add(i)
    all_content = texts + units + st.images + ([st.hero] if st.hero else [])
    bottom_most = max((b.bbox.y1 for b in all_content), default=0)
    if texts:
        hl = max(texts, key=lambda b: (b.el["font_size"], b.el.get("font_weight") == "bold", b.bbox.area))
        hl.role = "headline"
        st.headline = hl
        hfs = hl.el["font_size"]
        for b in texts:
            if b is hl:
                continue
            fs = b.el["font_size"]
            is_last = b.bbox.y1 >= bottom_most - 1
            long = len(str(b.el.get("content", "")).split()) >= 6
            if fs <= 0.4 * hfs and (long or is_last):
                b.role = "fine"
                st.fine.append(b)
            else:
                b.role = "body"
                st.body.append(b)

    # Units: rotated or corner-anchored ones are accents; largest remaining
    # text-bearing unit is the call to action; the rest are badges.
    flow_units = []
    for u in units:
        c = _corner_of(u.footprint, W, H)
        small = u.footprint.area < 0.08 * W * H
        if u.rotated or (c and small):
            u.role = "accent"
            u.corner = c or _nearest_corner(u.footprint, W, H)
            st.accents.append(u)
        else:
            flow_units.append(u)
    with_text = [u for u in flow_units if u.texts]
    if with_text:
        st.cta = max(with_text, key=lambda u: u.bbox.area)
        st.cta.role = "cta"
    for u in flow_units:
        if u is not st.cta:
            u.role = "badge"
            st.badges.append(u)
    for m in st.missing:
        m.corner = _nearest_corner(m.footprint, W, H)

    # Reading order: rows top-to-bottom (vertical overlap > 50%), then x.
    content = st.all_blocks()
    content.sort(key=lambda b: (b.footprint.cy, b.footprint.cx))
    rows: list[list[Block]] = []
    for b in content:
        fp = b.footprint
        if rows:
            r = rows[-1]
            rr = r[0].footprint
            ov = min(fp.y1, rr.y1) - max(fp.y0, rr.y0)
            if ov > 0.5 * min(fp.h, rr.h):
                r.append(b)
                continue
        rows.append([b])
    n = 0
    for ri, r in enumerate(rows):
        for b in sorted(r, key=lambda b: b.footprint.cx):
            b.order = n
            n += 1
            if b.role == "badge":
                b.row = ri
    return st

# ==========================================================================
# LAYOUT ARCHETYPES
#
# Layout archetypes.
#
# The target aspect ratio picks an arrangement a designer would use for that
# kind of format, and a single type-scale parameter `t` is searched so that
# everything fits:
#
#   stack  (portrait / square / story / skyscraper): vertical flow in source
#          reading order, logo and corner accents in a header band, fine
#          print pinned to the bottom.
#   split  (landscape, ~1.35-2.6:1): hero image in one column, a left-aligned
#          text column beside it.
#   row    (banner / leaderboard, > 2.6:1): left-to-right zones -
#          [logo] [hero] [text] [badges / accents] [CTA].
#
# Blocks are sized by their *ink* footprint (opaque pixels), not their box,
# so transparent padding in cut-out images never pushes content apart; the
# box is still kept fully on canvas.
# ==========================================================================

INF = 1e9
LAST_FAIL: list[str] = []  # why the most recent attempt was infeasible (diagnostics)


def _fail() -> None:
    f = sys._getframe(1)
    LAST_FAIL[:] = [f"{f.f_code.co_name}:{f.f_lineno}"]
    return None


# ---------------------------------------------------------------- context

@dataclass
class Ctx:
    W: float
    H: float
    st: Structure
    tight: float = 1.0  # < 1: compact margins and gaps (small ads, dense designs)

    @property
    def S(self) -> float:
        return min(self.W, self.H)

    @property
    def small(self) -> bool:
        """Display-ad sized (IAB units: 300x250, 728x90, 160x600, 970x250,
        all with a short side <= 300 px): space is so tight that compact
        spacing and side-by-side tricks beat smaller type. Social banners
        and covers keep their generous spacing."""
        return self.S <= 300

    @property
    def area_scale(self) -> float:
        return math.sqrt(self.W * self.H / (self.st.W * self.st.H))

    @property
    def m(self) -> float:
        return max(5.0, 0.055 * self.S * self.tight)

    @property
    def mx(self) -> float:
        return max(self.m, (0.03 if self.W / self.H < 5 else 0.02) * self.W * self.tight)

    @property
    def my(self) -> float:
        return max(self.m, 0.03 * self.H * self.tight) if self.W / self.H < 2.6 else self.m

    @property
    def story(self) -> bool:
        """Full-screen phone formats (9:16 story / reel) have platform UI
        over the top and bottom of the frame."""
        return 0.5 <= self.W / self.H <= 0.6 and self.H >= 1200

    @property
    def top_inset(self) -> float:
        return max(self.my, 0.10 * self.H) if self.story else self.my

    @property
    def bottom_inset(self) -> float:
        return max(self.my, 0.12 * self.H) if self.story else self.my

    def floor(self, role: str, k: float = 1.0) -> int:
        """Minimum legible font size for a role on this canvas. `k` < 1
        relaxes secondary copy when nothing fits; badges (lowest priority)
        relax furthest, the headline and CTA label least."""
        S = self.S
        base = {
            "headline": max(12.0, 0.028 * S),
            "label": max(8.0, 0.014 * S),  # CTA
            "body": max(9.0, 0.016 * S),
            "badge": max(8.0, 0.014 * S),
            "fine": max(7.0, 0.010 * S),
        }[role]
        if role == "headline":
            return int(round(base * max(k, 0.75)))
        if role == "label":
            return max(7 if k >= 0.45 else 5, int(round(base * max(k, 0.85 if k >= 0.45 else k))))
        hard_min = 6 if k >= 0.5 else 5
        if role == "badge":  # last resort on tiny strips: tags become decorative chips
            hard_min = 4 if k >= 0.4 else 3
        return max(hard_min, int(round(base * k)))


# ---------------------------------------------------------------- placement

@dataclass
class Placed:
    block: Block
    fp: Rect  # ink footprint on the target canvas


@dataclass
class Layout:
    ctx: Ctx
    out: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> new fields
    placed: list[Placed] = field(default_factory=list)
    text_ratio: list[float] = field(default_factory=list)  # placed fs / source-scaled fs
    t: float = 0.0
    k: float = 1.0
    name: str = ""
    sx: dict[int, float] = field(default_factory=dict)  # per-block horizontal pill squeeze
    stickers: set[int] = field(default_factory=set)  # accents deliberately layered on the hero

    # A rigid block maps source -> target by a similarity transform anchored
    # on its ink footprint: groups keep their internal layout exactly.
    def place_rigid(self, b: Block, u: float, fx: float, fy: float) -> None:
        fp = b.footprint
        ext = extent(b)
        ux = u * self.sx.get(id(b), 1.0)
        # keep the whole box on canvas
        x0 = fx + (ext.x0 - fp.x0) * ux
        y0 = fy + (ext.y0 - fp.y0) * u
        x1, y1 = x0 + ext.w * ux, y0 + ext.h * u
        dx = -x0 if x0 < 0 else (self.ctx.W - x1 if x1 > self.ctx.W else 0)
        dy = -y0 if y0 < 0 else (self.ctx.H - y1 if y1 > self.ctx.H else 0)
        fx, fy = fx + dx, fy + dy
        for e in b.els:
            g = {
                "x": fx + (e["x"] - fp.x0) * ux,
                "y": fy + (e["y"] - fp.y0) * u,
                "width": e["width"] * ux,
                "height": e["height"] * u,
            }
            if e["type"] == "text":
                if ux < u:  # squeezed pill: label spans the new pill width
                    g["x"], g["width"] = fx, fp.w * ux
                g.update(self._unit_label(b, e, u, g))
            self.out[e["id"]] = g
        self.placed.append(Placed(b, Rect(fx, fy, fx + fp.w * ux, fy + fp.h * u)))

    def _unit_label(self, b: Block, e: dict, u: float, g: dict) -> dict:
        """Scaled label: integer size that still wraps like the source and
        fits inside its pill, centred on it (optically, by cap height, for a
        single line)."""
        shapes = [s for s in b.els if s["type"] == "shape" and box(s).inter(box(e)) >= 0.9 * box(e).area]
        pill = shapes[0] if len(shapes) == 1 and not (e.get("rotation") or 0) else None
        limit_h = pill["height"] * u if pill else INF
        src = measure(e, int(e["font_size"]), e["width"])
        fs = max(1, int(math.floor(e["font_size"] * u + 1e-6)))
        f = measure(e, fs, g["width"])
        while fs > 1 and (f.lines > src.lines or f.width > g["width"] or f.height > limit_h):
            fs -= 1
            f = measure(e, fs, g["width"])
        self.text_ratio.append(fs / max(1e-6, e["font_size"] * self.t))
        res = {"font_size": fs, "height": max(g["height"], f.height)}
        if pill:
            sy = (pill["y"] - e["y"]) * u + g["y"]
            if f.lines == 1:
                asc, desc, cap = cap_metrics(e, fs)
                res["y"] = sy + limit_h / 2 - asc + cap / 2
            else:
                res["y"] = sy + (limit_h - f.height) / 2
            res["height"] = f.height
        return res

    def shift(self, pl: Placed, dx: float) -> None:
        for e in pl.block.els:
            self.out[e["id"]]["x"] += dx
        pl.fp = pl.fp.moved(dx, 0)

    def place_text(self, b: Block, f: TextFit, x: float, y: float, align: str) -> None:
        e = b.el
        w = f.box_width()
        self.out[e["id"]] = {
            "x": x, "y": y, "width": w, "height": math.ceil(f.height),
            "font_size": f.font_size, "text_align": align,
        }
        self.text_ratio.append(f.font_size / max(1e-6, e["font_size"] * self.t))
        lx = x
        if align == "center":
            lx = x + (w - f.width) / 2
        elif align == "right":
            lx = x + w - f.width
        self.placed.append(Placed(b, Rect(lx, y, lx + f.width, y + f.height)))

    def place_box(self, e: dict, x: float, y: float, w: float, h: float) -> None:
        self.out[e["id"]] = {"x": x, "y": y, "width": w, "height": h}


def extent(b: Block) -> Rect:
    """Union of members' (rotated) boxes: what must stay on canvas."""
    r = rbox(b.els[0])
    for e in b.els[1:]:
        r = r.union(rbox(e))
    return r


def pill_sx(b: Block) -> float:
    """Horizontal squeeze for a button-like unit (one shape, its label
    inside): trims side padding down to a comfortable minimum so buttons
    don't eat strip width. 1.0 for anything else."""
    shapes = [e for e in b.els if e["type"] == "shape"]
    if len(shapes) != 1 or not b.texts or b.rotated or len(b.els) != 1 + len(b.texts):
        return 1.0
    s = shapes[0]
    widest = 0.0
    for t in b.texts:
        if box(t).inter(box(s)) < 0.9 * box(t).area:
            return 1.0
        widest = max(widest, measure(t, int(t["font_size"]), t["width"]).width)
    fs = max(t["font_size"] for t in b.texts)
    need = widest + 2 * max(0.45 * s["height"], 0.9 * fs)
    return min(1.0, need / max(s["width"], 1e-6))


def rigid_max_scale(b: Block, ctx: Ctx) -> float:
    ext = extent(b)
    return min(ctx.W / max(ext.w, 1e-6), ctx.H / max(ext.h, 1e-6))


def label_min_scale(b: Block, ctx: Ctx, k: float) -> float:
    fl = ctx.floor("badge" if b.role in ("badge", "accent") else "label", k)
    return max([fl / t["font_size"] for t in b.texts] or [0.0])


# ---------------------------------------------------------------- text sizing

def text_fs_target(b: Block, ctx: Ctx, t: float, k: float, cap: Optional[int] = None) -> tuple[int, int]:
    """(preferred size, floor) for a free text block at type scale t."""
    role = "headline" if b.role == "headline" else "fine" if b.role == "fine" else "body"
    fl = ctx.floor(role, k)
    want = max(fl, int(round(b.el["font_size"] * t)))
    if cap is not None:
        want = min(want, cap)
    return want, min(fl, want)


def fit_text(b: Block, ctx: Ctx, t: float, k: float, max_w: float, max_h: float = INF,
             cap: Optional[int] = None) -> Optional[TextFit]:
    want, fl = text_fs_target(b, ctx, t, k, cap)
    f = fit(b.el, max_w, max_h, want, fl)
    if f is None:
        return _fail()
    return balance(b.el, f, max_w)


def text_tiers(st: Structure) -> list[Block]:
    return ([st.headline] if st.headline else []) + st.body + st.fine


# ---------------------------------------------------------------- shared bits

def place_background_and_deco(L: Layout) -> None:
    ctx, st = L.ctx, L.ctx.st
    for b in st.background:
        L.place_box(b.el, 0, 0, ctx.W, ctx.H)
    # Decorative shapes re-anchor to the corner/edge they came from and
    # scale with the canvas' short side; they may overlap content (they are
    # low-opacity layering). They stay on canvas, unless they bled off it in
    # the source: then the same share of them bleeds off the same edge.
    s = min(ctx.S / min(st.W, st.H), max(ctx.W / st.W, ctx.H / st.H))
    for b in st.deco:
        e = b.el
        w = e["width"] * s
        h = e["height"] * s
        k = min(1.0, 0.9 * ctx.S / max(w, h, 1e-6)) if e.get("shape_type") != "line" else 1.0
        w, h = w * k, h * k
        w, h = min(w, ctx.W), min(h, ctx.H)
        fx, fy = (e["x"] + e["width"] / 2) / st.W, (e["y"] + e["height"] / 2) / st.H
        x = min(max(0, fx * ctx.W - w / 2), ctx.W - w)
        y = min(max(0, fy * ctx.H - h / 2), ctx.H - h)
        if bleeds(e, {"width": st.W, "height": st.H}):
            r = rbox(e)
            over = lambda a: min(0.5, max(0.0, a))  # never more than half off
            if r.x0 < 0:
                x = -over(-r.x0 / r.w) * w
            elif r.x1 > st.W:
                x = ctx.W - w + over((r.x1 - st.W) / r.w) * w
            if r.y0 < 0:
                y = -over(-r.y0 / r.h) * h
            elif r.y1 > st.H:
                y = ctx.H - h + over((r.y1 - st.H) / r.h) * h
        L.place_box(e, x, y, w, h)


def place_panels(L: Layout) -> None:
    """Fit each panel around wherever its texts landed, with the source's
    padding scaled like the type. Padding gives way before it would run
    under other content; the panel always stays on canvas."""
    ctx, st = L.ctx, L.ctx.st
    for pb in st.panels:
        e = pb.el
        ids = {t["id"] for t in pb.holds}
        mine = [p for p in L.placed if p.block.el["id"] in ids]
        if len(mine) != len(ids):  # texts not laid out: keep the panel's relative spot
            sx, sy = ctx.W / st.W, ctx.H / st.H
            L.place_box(e, e["x"] * sx, e["y"] * sy, e["width"] * sx, e["height"] * sy)
            continue
        inner = mine[0].fp
        for p in mine[1:]:
            inner = inner.union(p.fp)
        src = box(pb.holds[0])
        for t in pb.holds[1:]:
            src = src.union(box(t))
        sb = box(e)
        u = sum(L.out[t["id"]]["font_size"] / t["font_size"] for t in pb.holds) / len(pb.holds)
        pad = {"l": (src.x0 - sb.x0) * u, "t": (src.y0 - sb.y0) * u,
               "r": (sb.x1 - src.x1) * u, "b": (sb.y1 - src.y1) * u}
        for p in L.placed:
            if p.block.role in NON_CONTENT or p in mine:
                continue
            o = p.fp
            if o.x1 > inner.x0 and o.x0 < inner.x1:  # above or below the copy
                if o.y1 <= inner.y0:
                    pad["t"] = min(pad["t"], inner.y0 - o.y1 - 0.25 * ctx.m)
                elif o.y0 >= inner.y1:
                    pad["b"] = min(pad["b"], o.y0 - inner.y1 - 0.25 * ctx.m)
            if o.y1 > inner.y0 and o.y0 < inner.y1:  # beside it
                if o.x1 <= inner.x0:
                    pad["l"] = min(pad["l"], inner.x0 - o.x1 - 0.25 * ctx.m)
                elif o.x0 >= inner.x1:
                    pad["r"] = min(pad["r"], o.x0 - inner.x1 - 0.25 * ctx.m)
        x0 = max(0.0, inner.x0 - max(0.0, pad["l"]))
        y0 = max(0.0, inner.y0 - max(0.0, pad["t"]))
        x1 = min(ctx.W, inner.x1 + max(0.0, pad["r"]))
        y1 = min(ctx.H, inner.y1 + max(0.0, pad["b"]))
        g = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
        if e.get("corner_radius"):
            g["corner_radius"] = min(e["corner_radius"] * u, (y1 - y0) / 2)
        L.out[e["id"]] = g


def place_corner(L: Layout, b: Block, u: float, corner: str, inset_x: float, inset_y: float) -> Rect:
    fp = b.footprint
    w, h = fp.w * u, fp.h * u
    x = inset_x if corner[1] == "l" else L.ctx.W - inset_x - w
    y = inset_y if corner[0] == "t" else L.ctx.H - inset_y - h
    L.place_rigid(b, u, x, y)
    return L.placed[-1].fp


def accent_scale(a: Block, ctx: Ctx, t: float, k: float, cap_w: float, cap_h: float = INF,
                 cta_u: Optional[float] = None) -> Optional[float]:
    """Scale for a corner accent: at least legible, never larger than the
    CTA (a tag never outranks the button), within the given caps."""
    umin = label_min_scale(a, ctx, k)
    fp = a.footprint
    u = max(t, umin)
    if cta_u:
        u = min(u, max(cta_u, umin))
    u = min(u, cap_w / fp.w, cap_h / fp.h, rigid_max_scale(a, ctx))
    return u if u >= umin * 0.999 else None


def corner_insets(ctx: Ctx, corner: str) -> tuple[float, float]:
    iy = ctx.my * 0.6
    if ctx.story:
        iy = ctx.top_inset if corner[0] == "t" else ctx.bottom_inset
    return ctx.mx * 0.6, iy


def place_stickers(L: Layout, t: float, k: float, cta_u: Optional[float] = None) -> bool:
    """Fallback for accents that can't get a corner: layer each one on a
    corner of the product, like a sticker on packaging, covering at most a
    quarter of the product's ink. Returns False if there is no product."""
    st, ctx = L.ctx.st, L.ctx
    hp = next((p for p in L.placed if p.block is st.hero), None)
    if hp is None:
        return False
    corners = ["tr", "tl", "br", "bl"]
    for i, a in enumerate(st.accents):
        c = a.corner or "tr"
        c = c if c not in [x.block.corner for x in L.placed if id(x.block) in L.stickers] else corners[i % 4]
        u = accent_scale(a, ctx, t, k, cap_w=0.9 * hp.fp.w + 0.5 * ctx.W, cta_u=cta_u)
        if u is None:
            return False
        w, h = a.footprint.w * u, a.footprint.h * u
        f = min(0.4, math.sqrt(0.25 * hp.fp.area / max(w * h, 1e-6)))  # overlap fraction
        x = hp.fp.x1 - f * w if c[1] == "r" else hp.fp.x0 - (1 - f) * w
        y = hp.fp.y0 - (1 - f) * h if c[0] == "t" else hp.fp.y1 - f * h
        x = min(max(x, 0.0), ctx.W - w)
        y = min(max(y, 0.0), ctx.H - h)
        L.place_rigid(a, u, x, y)
        L.stickers.add(id(a))
    return True


def place_missing(L: Layout, b: Block, u: float) -> None:
    """Put a missing-asset placeholder in the free corner nearest its
    source corner, shrinking it until it touches no content."""
    ctx = L.ctx
    content = [p.fp for p in L.placed if p.block.role not in ("background", "deco")]
    src = b.corner or "br"
    corners = sorted(["br", "bl", "tr", "tl"], key=lambda c: (c[0] != src[0]) * 2 + (c[1] != src[1]))
    inset = max(2.0, 0.25 * ctx.m)
    fp = b.footprint
    best = None
    for shrink in (1.0, 0.75, 0.5, 0.35):
        for c in corners:
            w, h = fp.w * u * shrink, fp.h * u * shrink
            x = inset if c[1] == "l" else ctx.W - inset - w
            y = inset if c[0] == "t" else ctx.H - inset - h
            r = Rect(x, y, x + w, y + h)
            hit = sum(r.inter(o) for o in content)
            if best is None or hit < best[0] - 1e-6:
                best = (hit, u * shrink, x, y)
            if hit == 0:
                break
        if best[0] == 0:
            break
    _, uu, x, y = best
    L.place_rigid(b, uu, x, y)


def missing_scale(b: Block, ctx: Ctx, t: float) -> float:
    """A missing asset renders as a grey placeholder: keep it small and out
    of the way, never over content."""
    fp = b.footprint
    u = t * 0.5
    u = min(u, 0.07 * ctx.H / max(fp.h, 1e-6), 0.10 * ctx.W / max(fp.w, 1e-6))
    return min(u, rigid_max_scale(b, ctx))


def source_alignment(st: Structure) -> str:
    texts = text_tiers(st)
    if not texts:
        return "center"
    votes = [b.el.get("text_align") or "left" for b in texts]
    centered = sum(abs(b.footprint.cx - st.W / 2) < 0.08 * st.W for b in texts)
    if votes.count("center") * 2 >= len(votes) or centered * 2 > len(texts):
        return "center"
    return max(set(votes), key=votes.count)


# ---------------------------------------------------------------- stack

def stack(ctx: Ctx, t: float, k: float, sticker: bool = False, beside: bool = False) -> Optional[Layout]:
    """`beside=True` sets the content badges as a column beside the product
    (a narrow product leaves the width either side of it empty), which
    frees a whole row of the stack on small ads."""
    st = ctx.st
    if beside and (not ctx.small or not st.badges or st.hero is None):
        return _fail()  # nothing to gain over the plain stack
    W, H, mx = ctx.W, ctx.H, ctx.mx
    cw = W - 2 * mx
    g = max(ctx.m * 0.6, 0.022 * H * ctx.tight)
    L = Layout(ctx, t=t, name="stack")
    align = source_alignment(st)
    place_background_and_deco(L)

    # Header band: logo (if it opens the design) + top-corner accents.
    logo_pinned = st.logo is not None and st.logo.footprint.cy < 0.35 * st.H
    header_h = 0.0
    logo_u = 0.0
    if logo_pinned:
        fp = st.logo.footprint
        logo_u = min(t, 0.13 * H / fp.h, 0.5 * cw / fp.w, rigid_max_scale(st.logo, ctx))
        header_h = fp.h * logo_u
    logo_w = st.logo.footprint.w * logo_u if logo_pinned else 0.0

    # Accents stay in their source corner. Beside a centred logo if there
    # is room; otherwise they keep the corner and the logo drops below them
    # (an accent never enters the logo -> headline -> CTA reading line).
    corner_acc = [] if sticker else st.accents
    top_acc = [a for a in corner_acc if (a.corner or "t")[0] == "t"]
    bot_acc = [a for a in corner_acc if (a.corner or "t")[0] == "b"]
    acc_u: dict[int, float] = {}
    above_h = 0.0  # top accents that don't fit beside the logo get their own band
    for a in top_acc + bot_acc:
        fp = a.footprint
        u = accent_scale(a, ctx, t, k, 0.3 * W)
        if a in top_acc and logo_pinned and align == "center":
            room = (W - logo_w) / 2 - mx - g  # space beside a centred logo
            u = accent_scale(a, ctx, t, k, min(0.3 * W, room)) if room > 0 else None
            if u is None:
                u = accent_scale(a, ctx, t, k, 0.6 * cw)
                if u is None:
                    return _fail()
                above_h = max(above_h, fp.h * u)
        if u is None:
            u = accent_scale(a, ctx, t, k, 0.6 * cw)
            if u is None:
                return _fail()
        acc_u[id(a)] = u
        if a in top_acc:
            header_h = max(header_h, fp.h * u)
    logo_y_off = 0.0
    if above_h:
        logo_h = st.logo.footprint.h * logo_u if logo_pinned else 0.0
        logo_y_off = above_h + 0.5 * g
        header_h = logo_y_off + logo_h

    # Footer: fine print + bottom-corner items (missing assets, accents).
    miss_u = {id(b): missing_scale(b, ctx, t) for b in st.missing}
    corner_w = 0.0
    for b in st.missing + bot_acc:
        u = miss_u[id(b)] if id(b) in miss_u else acc_u[id(b)]
        if (b.corner or "br")[0] == "b":
            corner_w = max(corner_w, b.footprint.w * u)
    fine_w = cw - 2 * (corner_w + g) if corner_w else cw
    fine_fits = []
    for b in st.fine:
        f = fit_text(b, ctx, t, k, fine_w)
        if f is None:
            return _fail()
        fine_fits.append((b, f))
    footer_h = sum(f.height for _, f in fine_fits) + g * 0.3 * max(0, len(fine_fits) - 1)
    for b in st.missing + bot_acc:
        u = miss_u[id(b)] if id(b) in miss_u else acc_u[id(b)]
        footer_h = max(footer_h, b.footprint.h * u) if (b.corner or "br")[0] == "b" else footer_h

    # Flow items in reading order.
    items: list[tuple[str, Any, float]] = []  # (kind, payload, height)
    flow = [b for b in st.flow() if b is not st.logo]
    if st.logo and not logo_pinned:
        flow.append(st.logo)
    flow.sort(key=lambda b: b.order)
    # Narrow canvases: tighten button padding only where a pill would
    # otherwise not fit the column at its minimum legible label size.
    for b in st.badges + ([st.cta] if st.cta else []):
        if b.footprint.w * label_min_scale(b, ctx, k) > 0.7 * cw:
            L.sx[id(b)] = pill_sx(b)
    sxw = lambda b: b.footprint.w * L.sx.get(id(b), 1.0)  # noqa: E731
    hl_fs: Optional[int] = None
    seen_rows: set[int] = set()
    for b in flow:
        if b.role in ("headline", "body"):
            cap = None if b.role == "headline" or hl_fs is None else max(1, int(hl_fs * 0.75))
            src_w = b.el["width"] * max(t, W / st.W)
            f = fit_text(b, ctx, t, k, min(cw, max(src_w, 0.6 * cw)), cap=cap)
            if f is None:
                return _fail()
            if b.role == "headline":
                hl_fs = f.font_size
            items.append(("text", (b, f), f.height))
        elif b.role == "hero":
            items.append(("hero", b, 0.0))
        elif b.role == "badge":
            if beside or b.row in seen_rows:
                continue
            seen_rows.add(b.row)
            row = sorted([x for x in st.badges if x.row == b.row], key=lambda x: x.footprint.cx)
            rows = badge_rows(row, ctx, t, k, cw, g * 0.5, sxw)
            if rows is None:
                return _fail()
            items.append(("badges", rows, rows_height(rows, g * 0.5)))
        else:  # cta, logo-in-flow, other images
            fp = b.footprint
            u = min(t, cw / sxw(b), rigid_max_scale(b, ctx))
            if b.role == "cta":
                u = min(max(u, label_min_scale(b, ctx, k)), cw / sxw(b))
                if u < label_min_scale(b, ctx, k) * 0.999:
                    return _fail()
            items.append(("rigid", (b, u), fp.h * u))

    top = ctx.top_inset + (header_h + g if header_h else 0)
    bottom = H - ctx.bottom_inset - (footer_h + g if footer_h else 0)
    avail = bottom - top
    gaps = []
    for i in range(1, len(items)):
        both_text = items[i][0] == "text" and items[i - 1][0] == "text"
        gaps.append(g * (0.45 if both_text else 1.0))
    fixed = sum(h for _, _, h in items) + sum(gaps)

    # Badge column beside the product: at the unrelaxed legible size if it
    # fits, never above the CTA's label size (tags don't outrank the button).
    side = None
    if beside:
        if not any(kind == "hero" for kind, _, _ in items):
            return _fail()
        tags = sorted(st.badges, key=lambda b: b.order)
        u_lo = max(label_min_scale(b, ctx, k) for b in tags)
        u_b = max(t, max(label_min_scale(b, ctx, 1.0) for b in tags))
        u_b = max(u_lo, min(u_b, 0.6 * cw / max(sxw(b) for b in tags)))
        tag_fs = max(e["font_size"] for b in tags for e in b.texts)
        ci = next((i for i, (kind, p, _) in enumerate(items) if kind == "rigid" and p[0] is st.cta), None)
        if ci is not None:  # the button stays a step above the tags: grow it, else cap them
            cfs = max(e["font_size"] for e in st.cta.texts)
            u_c = items[ci][1][1]
            need = 1.15 * u_b * tag_fs / cfs
            if need > u_c:
                u_c2 = min(need, cw / sxw(st.cta), rigid_max_scale(st.cta, ctx))
                if u_c2 > u_c:
                    fixed += st.cta.footprint.h * (u_c2 - u_c)
                    items[ci] = ("rigid", (st.cta, u_c2), st.cta.footprint.h * u_c2)
                    u_c = u_c2
            u_b = max(u_lo, min(u_b, u_c * cfs / 1.15 / tag_fs))
        sg = g * 0.5
        col_w = max(sxw(b) for b in tags) * u_b
        col_h = sum(b.footprint.h for b in tags) * u_b + sg * (len(tags) - 1)
        if col_w > 0.6 * cw + 0.5:
            return _fail()
        side = (tags, u_b, col_w, col_h, sg)

    hero_h = 0.0
    if st.hero is not None and any(kind == "hero" for kind, _, _ in items):
        fp = st.hero.footprint
        room_w = cw - (side[2] + g if side else 0.0)
        hmax = min(fp.h * t * 1.25, room_w * fp.h / fp.w, rigid_max_scale(st.hero, ctx) * fp.h, 0.5 * H)
        hmin = min(hmax, 0.14 * H)
        if avail - fixed < max(hmin, side[3] if side else 0.0):
            return _fail()
        hero_h = min(hmax, avail - fixed)
        pair_h = max(hero_h, side[3]) if side else hero_h
        items = [(kd, p, pair_h if kd == "hero" else h) for kd, p, h in items]
        fixed += pair_h
    if fixed > avail + 0.5:
        return _fail()

    # Spread leftover: grow gaps (up to 2.2x), then centre the block.
    left = avail - fixed
    if gaps:
        grow = min(left / len(gaps), g * 1.2)
        gaps = [x + grow * (0.5 if x < g else 1.0) for x in gaps]
        left = avail - sum(h for _, _, h in items) - sum(gaps)
    y = top + left * 0.45  # optical centre sits slightly above geometric

    def xpos(w: float) -> float:
        if align == "center":
            return (W - w) / 2
        if align == "right":
            return W - mx - w
        return mx

    for i, (kind, p, h) in enumerate(items):
        if i:
            y += gaps[i - 1]
        if kind == "text":
            b, f = p
            L.place_text(b, f, xpos(f.box_width()), y, align)
        elif kind == "hero":
            fp = st.hero.footprint
            u = hero_h / fp.h
            if side:
                tags, u_b, col_w, col_h, sg = side
                x0 = xpos(fp.w * u + g + col_w)
                L.place_rigid(st.hero, u, x0, y + (h - hero_h) / 2)
                yb = y + (h - col_h) / 2
                for b in tags:
                    L.place_rigid(b, u_b, x0 + fp.w * u + g + (col_w - sxw(b) * u_b) / 2, yb)
                    yb += b.footprint.h * u_b + sg
            else:
                L.place_rigid(st.hero, u, xpos(fp.w * u), y)
        elif kind == "badges":
            place_badge_rows(L, p, y, g * 0.5, xpos)
        else:
            b, u = p
            L.place_rigid(b, u, xpos(sxw(b) * u), y)
        y += h

    # Header and footer.
    if logo_pinned:
        lh = st.logo.footprint.h * logo_u
        ly = ctx.top_inset + (logo_y_off if above_h else (header_h - lh) / 2)
        L.place_rigid(st.logo, logo_u, xpos(logo_w), ly)
    for a in top_acc + bot_acc:
        c = a.corner or "tr"
        place_corner(L, a, acc_u[id(a)], c, *corner_insets(ctx, c))
    if sticker and not place_stickers(L, t, k):
        return _fail()
    fy = H - ctx.bottom_inset - sum(f.height for _, f in fine_fits) - g * 0.3 * max(0, len(fine_fits) - 1)
    for b, f in fine_fits:
        L.place_text(b, f, (W - f.box_width()) / 2 if align == "center" else xpos(f.box_width()), fy, align)
        fy += f.height + g * 0.3
    for b in st.missing:
        place_missing(L, b, miss_u[id(b)])
    return L


def badge_rows(row: list[Block], ctx: Ctx, t: float, k: float, max_w: float, gap: float,
               wf=lambda b: b.footprint.w):
    """Lay a row of badges out in as few lines as possible at scale <= t."""
    umin = max(label_min_scale(b, ctx, k) for b in row)
    for n_lines in range(1, len(row) + 1):
        per = math.ceil(len(row) / n_lines)
        lines = [row[i:i + per] for i in range(0, len(row), per)]
        widest = max(sum(wf(b) for b in ln) for ln in lines)
        gaps = gap * (per - 1)
        u = min(max(t, umin), (max_w - gaps) / widest)
        if u >= umin * 0.999 or n_lines == len(row):
            if u < umin * 0.999:
                return _fail()
            return [(ln, u) for ln in lines]
    return _fail()


def rows_height(rows, gap: float) -> float:
    return sum(max(b.footprint.h for b in ln) * u for ln, u in rows) + gap * (len(rows) - 1)


def place_badge_rows(L: Layout, rows, y: float, gap: float, xpos) -> None:
    for ln, u in rows:
        wf = lambda b: b.footprint.w * L.sx.get(id(b), 1.0)  # noqa: E731
        w = sum(wf(b) * u for b in ln) + gap * (len(ln) - 1)
        rh = max(b.footprint.h for b in ln) * u
        x = xpos(w)
        for b in ln:
            L.place_rigid(b, u, x, y + (rh - b.footprint.h * u) / 2)
            x += wf(b) * u + gap
        y += rh + gap


# ---------------------------------------------------------------- split

def split(ctx: Ctx, t: float, k: float, sticker: bool = False) -> Optional[Layout]:
    st = ctx.st
    if st.hero is None:
        return _fail()
    W, H, mx, my = ctx.W, ctx.H, ctx.mx, ctx.my
    g = max(ctx.m * 0.6, 0.03 * H * ctx.tight)
    L = Layout(ctx, t=t, name="split")
    place_background_and_deco(L)

    # Accents keep their source corner; whichever column owns that corner
    # gives up the band the accent needs.
    acc: list[tuple[Block, float, str]] = []
    res = {"tl": 0.0, "tr": 0.0, "bl": 0.0, "br": 0.0}
    for a in ([] if sticker else st.accents):
        c = a.corner or "tl"
        u = accent_scale(a, ctx, t, k, 0.22 * W, 0.25 * H)
        if u is None:
            return _fail()
        acc.append((a, u, c))
        res[c] = max(res[c], a.footprint.h * u + 0.5 * g)

    hero = st.hero
    fp = hero.footprint
    hy0, hy1 = my + res["tl"], H - my - res["bl"]
    u_h = min((hy1 - hy0) / fp.h, 0.46 * W / fp.w, rigid_max_scale(hero, ctx))
    hw = fp.w * u_h
    zone_w = max(hw, 0.36 * W)
    L.place_rigid(hero, u_h, mx + (zone_w - hw) / 2, hy0 + (hy1 - hy0 - fp.h * u_h) / 2)

    x0 = mx + zone_w + 1.5 * g
    cw = W - mx - x0
    if cw < 0.3 * W:
        return _fail()

    col: list[tuple[str, Any, float]] = []
    if st.logo:
        lfp = st.logo.footprint
        u = min(t, 0.16 * H / lfp.h, 0.6 * cw / lfp.w, rigid_max_scale(st.logo, ctx))
        col.append(("rigid", (st.logo, u), lfp.h * u))
    hl_fs = None
    seen_rows: set[int] = set()
    for b in [b for b in st.flow() if b is not hero]:
        if b.role in ("headline", "body"):
            cap = None if b.role == "headline" or hl_fs is None else max(1, int(hl_fs * 0.75))
            f = fit_text(b, ctx, t, k, cw, cap=cap)
            if f is None:
                return _fail()
            hl_fs = f.font_size if b.role == "headline" else hl_fs
            col.append(("text", (b, f), f.height))
        elif b.role == "badge":
            if b.row in seen_rows:
                continue
            seen_rows.add(b.row)
            row = sorted([x for x in st.badges if x.row == b.row], key=lambda x: x.footprint.cx)
            rows = badge_rows(row, ctx, t, k, cw, g * 0.4)
            if rows is None:
                return _fail()
            col.append(("badges", rows, rows_height(rows, g * 0.4)))
        else:
            bfp = b.footprint
            u = min(t, cw / bfp.w, rigid_max_scale(b, ctx))
            if b.role == "cta":
                u = min(max(u, label_min_scale(b, ctx, k)), 0.2 * H / bfp.h, cw / bfp.w)
                if u < label_min_scale(b, ctx, k) * 0.999:
                    return _fail()
            col.append(("rigid", (b, u), bfp.h * u))
    for b in st.fine:
        f = fit_text(b, ctx, t, k, cw)
        if f is None:
            return _fail()
        col.append(("text", (b, f), f.height))

    gaps = [g * (0.45 if col[i][0] == "text" and col[i - 1][0] == "text" else 1.0) for i in range(1, len(col))]
    total = sum(h for _, _, h in col) + sum(gaps)
    cy0, cy1 = my + res["tr"], H - my - res["br"]
    if total > cy1 - cy0 + 0.5:
        return _fail()
    y = cy0 + (cy1 - cy0 - total) / 2
    for i, (kind, p, h) in enumerate(col):
        if i:
            y += gaps[i - 1]
        if kind == "text":
            b, f = p
            L.place_text(b, f, x0, y, "left")
        elif kind == "badges":
            place_badge_rows(L, p, y, g * 0.4, lambda w: x0)
        else:
            b, u = p
            L.place_rigid(b, u, x0, y)
        y += h

    for a, u, c in acc:
        place_corner(L, a, u, c, mx * 0.5, my * 0.5)
    if sticker and not place_stickers(L, t, k):
        return _fail()
    for b in st.missing:
        place_missing(L, b, missing_scale(b, ctx, t))
    return L


# ---------------------------------------------------------------- row

def row(ctx: Ctx, t: float, k: float, inline: bool = False, sticker: bool = False) -> Optional[Layout]:
    """`inline=True` puts badges as a row under the copy instead of in their
    own zone: better when the text needs the width more than the tags do.
    Corner accents stay in their source corner: a left-corner accent gets a
    slim edge column before the logo/product; a right-corner accent sits in
    its corner above (or below) the CTA, which centres in the space left.
    `sticker=True` instead layers accents on the product."""
    st = ctx.st
    W, H, mx, my = ctx.W, ctx.H, ctx.mx, ctx.my
    ah = H - 2 * my
    r = W / H
    gz = max(ctx.m * 1.5, 0.025 * W * ctx.tight) if r < 5 else max(ctx.m, 0.016 * W * ctx.tight)  # gap between zones
    g = max(2.0, ah * 0.07)  # gap inside a zone
    L = Layout(ctx, t=t, name="row_inline" if inline else "row")
    place_background_and_deco(L)
    if r > 5:  # strips: tighten button padding
        for b in ([st.cta] if st.cta else []) + st.badges + st.accents:
            L.sx[id(b)] = pill_sx(b)
    sxw = lambda b: b.footprint.w * L.sx.get(id(b), 1.0)  # noqa: E731
    logo_in_text = st.logo is not None and r <= 5.0

    zones_left: list[tuple[Block, float]] = []
    if st.logo and not logo_in_text:
        fp = st.logo.footprint
        u = min(0.5 * ah / fp.h, 0.14 * W / fp.w, rigid_max_scale(st.logo, ctx))
        zones_left.append((st.logo, u))
    if st.hero:
        fp = st.hero.footprint
        u = min(ah / fp.h * (1.0 if r > 5 else 0.92), 0.25 * W / fp.w, rigid_max_scale(st.hero, ctx))
        zones_left.append((st.hero, u))
    for b in st.images:
        fp = b.footprint
        zones_left.append((b, min(0.8 * ah / fp.h, 0.15 * W / fp.w, rigid_max_scale(b, ctx))))

    # Right side: CTA (outermost), then badges/accents in a compact grid.
    cta_u = 0.0
    if st.cta:
        fp = st.cta.footprint
        lo = label_min_scale(st.cta, ctx, k)
        cap_h = (0.5 if r > 5 else 0.3) * ah
        if lo * fp.h > cap_h:  # tiny strips: a legible label beats a slim pill
            cap_h = 0.85 * ah
        cap_w = 0.22 * W if lo * sxw(st.cta) <= 0.22 * W else 0.4 * W
        cta_u = min(max(t * 1.1, lo), cap_h / fp.h, cap_w / sxw(st.cta))
        cta_u = min(cta_u, rigid_max_scale(st.cta, ctx))
        if cta_u < lo * 0.999:
            return _fail()
    # Content badges (in the source's reading flow) go in their own zone or
    # inline under the copy; accents never join them.
    extras = sorted(st.badges, key=lambda b: b.order)
    tags: list[Block] = []
    if inline:
        tags, extras = extras, []
    grid = None
    if extras:
        grid = badge_grid(extras, ctx, t, k, ah, (0.34 if k >= 0.4 else 0.42) * W, g, sxw)
        if grid is None:
            return _fail()
        if st.cta and grid[2] > cta_u:  # tags never outrank the button
            fp = st.cta.footprint
            cta_u = min(grid[2], cap_h / fp.h, cap_w / sxw(st.cta), rigid_max_scale(st.cta, ctx))

    left_acc: list[tuple[Block, float, str]] = []
    right_acc: list[tuple[Block, float, str]] = []
    for a in ([] if sticker else st.accents):
        c = a.corner or "tr"
        u = accent_scale(a, ctx, t, k, 0.15 * W, (0.45 if r > 5 else 0.35) * ah, cta_u or None)
        if u is None:
            return _fail()
        (left_acc if c[1] == "l" else right_acc).append((a, u, c))
    left_w = max([a.footprint.w * u for a, u, _ in left_acc] or [0.0])
    acc_rw = max([a.footprint.w * u for a, u, _ in right_acc] or [0.0])
    top_h = max([a.footprint.h * u for a, u, c in right_acc if c[0] == "t"] or [0.0])
    bot_h = max([a.footprint.h * u for a, u, c in right_acc if c[0] == "b"] or [0.0])
    band_y0, band_y1 = my + (top_h + g if top_h else 0), H - my - (bot_h + g if bot_h else 0)
    if st.cta and right_acc:  # the CTA centres in the space the corner accent leaves
        fp = st.cta.footprint
        cta_u = min(cta_u, (band_y1 - band_y0) / fp.h)
        if cta_u < lo * 0.999:
            return _fail()

    # Place left zones first (vertically centred): the box of a padded
    # image may need to shift right to stay on canvas, so the text column
    # starts after where they actually landed.
    # Only the first zone pays for transparent padding against the canvas
    # edge (later ones may overlap a neighbour's empty padding), so lead
    # with the one that wastes the least space.
    def pad(z):
        b, u = z
        return (b.footprint.x0 - extent(b).x0) * u
    if zones_left and pad(zones_left[0]) > mx:
        zones_left.sort(key=pad)
    x = mx + (left_w + gz if left_w else 0)
    for b, u in zones_left:
        fp = b.footprint
        L.place_rigid(b, u, x, (H - fp.h * u) / 2)
        x = L.placed[-1].fp.x1 + gz
    cell_w = max(sxw(st.cta) * cta_u if st.cta else 0.0, acc_rw)
    right_w = (cell_w + gz if cell_w else 0) + (grid[1] + gz if grid else 0)
    tx0 = x
    tw = W - mx - right_w - tx0
    if tw < 0.2 * W:
        return _fail()

    col: list[tuple[str, Any, float]] = []
    if logo_in_text:
        fp = st.logo.footprint
        u = min(t, 0.3 * ah / fp.h, 0.5 * tw / fp.w, rigid_max_scale(st.logo, ctx))
        col.append(("rigid", (st.logo, u), fp.h * u))
    def tag_rows(blocks: list[Block]):
        # a tag sits a step below the CTA in the hierarchy
        t_eb = t
        if st.cta:
            t_eb = min(t, 0.6 * st.cta.footprint.h * cta_u / max(e.footprint.h for e in blocks))
        return badge_rows(blocks, ctx, t_eb, k, tw, g, sxw)

    hl_fs = None
    for b in text_tiers(st) + ([None] if tags else []):
        if tags and (b is None or b.role == "fine"):  # tag row goes above the fine print
            rows = tag_rows(tags)
            if rows is None:
                return _fail()
            col.append(("badges", rows, rows_height(rows, g)))
            tags = []
        if b is None:
            break
        cap = None if b.role == "headline" or hl_fs is None else max(1, int(hl_fs * 0.75))
        f = fit_text(b, ctx, t, k, tw, ah, cap=cap)
        if f is None:
            return _fail()
        hl_fs = f.font_size if b.role == "headline" else hl_fs
        col.append(("text", (b, f), f.height))
    gaps = [g * (0.35 if col[i][0] == "text" and col[i - 1][0] == "text" else 0.8) for i in range(1, len(col))]
    total = sum(h for _, _, h in col) + sum(gaps)
    if total > ah + 0.5:
        return _fail()

    # Nothing on the right (no CTA button, no badges): centre the whole
    # lockup instead of leaving the right half of the banner empty.
    align = "left"
    if not st.cta and not grid and not right_acc:
        def item_w(kind, p):
            if kind == "text":
                return p[1].width
            if kind == "badges":
                return max(sum(b.footprint.w * u for b in ln) + g * (len(ln) - 1) for ln, u in p)
            return p[0].footprint.w * p[1]
        used_w = max(item_w(kind, p) for kind, p, _ in col)
        dx = (W - 2 * mx - (tx0 - mx) - used_w) / 2
        for pl in [pl for pl in L.placed if pl.block in [b for b, _ in zones_left]]:
            L.shift(pl, dx)
        if zones_left:
            tx0 += dx
            tw -= dx
            # narrow the boxes to the room left (same line breaks)
            for i, (kind, p, h) in enumerate(col):
                if kind == "text":
                    b, f = p
                    f2 = measure(b.el, f.font_size, tw)
                    if f2.lines == f.lines:
                        col[i] = (kind, (b, f2), h)
        else:  # a lone text lockup: centre it on the canvas, as a designer would
            align = source_alignment(st)
            tx0, tw = mx, W - 2 * mx

    def xpos(w: float) -> float:
        return tx0 + (tw - w) / 2 if align == "center" else tx0

    y = (H - total) / 2
    for i, (kind, p, h) in enumerate(col):
        if i:
            y += gaps[i - 1]
        if kind == "text":
            b, f = p
            L.place_text(b, f, xpos(f.box_width()), y, align)
        elif kind == "badges":
            place_badge_rows(L, p, y, g, xpos)
        else:
            b, u = p
            L.place_rigid(b, u, xpos(b.footprint.w * u), y)
        y += h
    xr = W - mx
    if st.cta:
        fp = st.cta.footprint
        w = sxw(st.cta) * cta_u
        L.place_rigid(st.cta, cta_u, xr - w, band_y0 + (band_y1 - band_y0 - fp.h * cta_u) / 2)
    xr -= cell_w + gz if cell_w else 0
    if grid:
        place_grid(L, grid, xr - grid[1], ah, my, g, sxw)
    for a, u, c in left_acc + right_acc:
        place_corner(L, a, u, c, *corner_insets(ctx, c))
    if sticker and not place_stickers(L, t, k, cta_u or None):
        return _fail()
    for b in st.missing:
        place_missing(L, b, missing_scale(b, ctx, t))
    return L


def badge_grid(blocks: list[Block], ctx: Ctx, t: float, k: float, ah: float, max_w: float, g: float,
               wf=lambda b: b.footprint.w):
    """Pack badges/accents into ncols columns; pick the layout with the
    largest scale that fits the zone height and width."""
    best = None
    umin = max(label_min_scale(b, ctx, k) for b in blocks)
    # Aim for the unrelaxed legible size when the zone has room for it,
    # even if the type scale is smaller: spare height is free here.
    want = max(t, umin, max(label_min_scale(b, ctx, 1.0) for b in blocks))
    for ncols in range(1, len(blocks) + 1):
        cols = [blocks[i::ncols] for i in range(ncols)]
        col_h = max(sum(b.footprint.h for b in c) for c in cols)
        n_rows = max(len(c) for c in cols)
        u = min(want, (ah - g * (n_rows - 1)) / col_h)
        widths = [max(wf(b) for b in c) for c in cols]
        width = sum(widths) * u + g * (ncols - 1)
        if width > max_w:
            u = min(u, (max_w - g * (ncols - 1)) / sum(widths))
            width = sum(widths) * u + g * (ncols - 1)
        u = min([u] + [rigid_max_scale(b, ctx) for b in blocks])
        if u >= umin * 0.999 and (best is None or u > best[2] * 1.05):
            best = (cols, width, u, widths)
    return best


def place_grid(L: Layout, grid, x0: float, ah: float, my: float, g: float, wf) -> None:
    cols, width, u, widths = grid
    x = x0
    for c, cw in zip(cols, widths):
        h = sum(b.footprint.h for b in c) * u + g * (len(c) - 1)
        y = my + (ah - h) / 2
        for b in c:
            fp = b.footprint
            L.place_rigid(b, u, x + (cw * u - wf(b) * u) / 2, y)
            y += fp.h * u + g
        x += cw * u + g


def _with_panels(fn: Callable[..., Optional[Layout]]) -> Callable[..., Optional[Layout]]:
    def solved(ctx: Ctx, t: float, k: float, **kw) -> Optional[Layout]:
        L = fn(ctx, t, k, **kw)
        if L is not None:
            place_panels(L)
        return L
    return solved


ARCHETYPES = {
    "stack": _with_panels(stack),
    "stack_beside": _with_panels(lambda ctx, t, k, sticker=False: stack(ctx, t, k, sticker=sticker, beside=True)),
    "split": _with_panels(split),
    "row": _with_panels(row),
    "row_inline": _with_panels(lambda ctx, t, k, sticker=False: row(ctx, t, k, inline=True, sticker=sticker)),
}

# ==========================================================================
# HARD-RULE CHECKS
#
# Hard-rule checks on an output design.
#
# [rule]   = stated in RULES.md / the schema / the FAQ.
# [policy] = our own stricter choice; reported separately so it never blocks
#            an otherwise valid layout.
# ==========================================================================

REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}
UNCHANGED_RULE = ["type", "content", "font_family", "asset_id"]
UNCHANGED_POLICY = ["font_weight", "font_style", "color", "fill_color", "z_index", "group_id", "rotation", "opacity"]
TOL = 0.5


def is_background(el: dict, canvas: dict) -> bool:
    return el["type"] == "image" and (
        el.get("fit") == "cover"
        or (el["width"] >= 0.9 * canvas["width"] and el["height"] >= 0.9 * canvas["height"])
    )


def check(source: dict[str, Any], output: dict[str, Any], target: dict[str, Any]) -> dict[str, list[str]]:
    rule: list[str] = []
    policy: list[str] = []
    out_els = output.get("elements", [])
    cw, ch = target["width"], target["height"]

    if output.get("canvas") != {"width": cw, "height": ch}:
        rule.append(f"canvas {output.get('canvas')} != target {cw}x{ch}")
    src_ids = Counter(e["id"] for e in source["elements"])
    out_ids = Counter(e.get("id") for e in out_els)
    if src_ids != out_ids:
        rule.append(f"id multiset mismatch: missing {sorted((src_ids - out_ids).elements())}, "
                    f"extra {sorted((out_ids - src_ids).elements())}")
    src = {e["id"]: e for e in source["elements"]}

    for e in out_els:
        eid = e.get("id")
        for f in REQUIRED_COMMON + REQUIRED_BY_TYPE.get(e.get("type"), []):
            if f not in e:
                rule.append(f"{eid}: missing field {f}")
        if eid not in src or any(f not in e for f in REQUIRED_COMMON):
            continue
        s = src[eid]
        for f in UNCHANGED_RULE:
            if s.get(f) != e.get(f):
                rule.append(f"{eid}: {f} changed")
        for f in UNCHANGED_POLICY:
            if s.get(f) != e.get(f):
                policy.append(f"{eid}: {f} changed")
        if e["width"] < 0 or e["height"] < 0:
            rule.append(f"{eid}: negative size")

        r = on_canvas_extent(e)  # strict: plain box and rotated bounds
        if r.x0 < 0 or r.y0 < 0 or r.x1 > cw or r.y1 > ch:
            msg = f"{eid}: off-canvas ({r.x0:.1f},{r.y0:.1f})-({r.x1:.1f},{r.y1:.1f})"
            if bleeds(s, source["canvas"]):
                policy.append(msg + " (intentional bleed)")
            else:
                rule.append(msg)

        if e["type"] == "image" and not is_background(s, source["canvas"]):
            a0 = s["width"] / max(s["height"], 1e-9)
            a1 = e["width"] / max(e["height"], 1e-9)
            if abs(a1 / a0 - 1) > 0.05:
                rule.append(f"{eid}: aspect {a1:.3f} vs source {a0:.3f}")

        if e["type"] == "text":
            fs = e.get("font_size", 0)
            if fs != int(fs):
                rule.append(f"{eid}: non-integer font_size {fs}")
            if fs >= 1:
                f = measure(e, int(fs), e["width"])
                if f.width > e["width"] + TOL:
                    rule.append(f"{eid}: line {f.width:.1f}px wider than box {e['width']:.1f}")
                need = f.lines * fs * line_height(e)
                if need > e["height"] + TOL:
                    rule.append(f"{eid}: {f.lines} lines need {need:.1f}px > box {e['height']:.1f}")
            if (e.get("letter_spacing") or 0) != 0:
                policy.append(f"{eid}: letter_spacing != 0 (renderer wraps without it)")

    # z-order: relative order of z_index preserved
    out = {e.get("id"): e for e in out_els}
    ids = [i for i in src if i in out]
    for a in ids:
        for b in ids:
            za, zb = src[a].get("z_index", 0), src[b].get("z_index", 0)
            if za < zb and not out[a].get("z_index", 0) < out[b].get("z_index", 0):
                policy.append(f"z-order {a} < {b} not preserved")
    return {"rule": rule, "policy": policy}
