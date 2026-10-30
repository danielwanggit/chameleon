# CHAMELEON: layout-aware resizing for Adobe Express designs

`resize(source, target_canvas)` takes a design and a target canvas and
**re-lays it out** as a designer would. It does not stretch. A square post
becomes a story, a banner, a leaderboard or a skyscraper. Every element is
still there, text stays readable, logos and products keep their proportions,
and accents stay on the edges of the frame.

All engine code is in **`engine/resize.py`**. Apart from this README and
our dev tools in `eval/`, the rest of the repo is the organizers' starter
scaffolding (loading, writing, rendering, Docker), unchanged.

```
input:  a task JSON (source design + target canvas)
              |
              v
        engine/resize.py   analyze -> lay out -> score -> check -> (fallback)
              |
              v
output: <task>_output.json  (schema-valid resized design)
        <task>_output.png   (rendered preview)
```

## Setup and running

Requires Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main          # processes every input/*.json -> output/
```

Docker (how stage-one scoring runs, fully offline):

```bash
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  chameleon-engine
```

## Model information

**The shipped engine uses no machine-learning model and no weights.** It is a
deterministic, CPU-only constraint layout engine. For the same input it always
produces the same output, it needs nothing at run time, and each task takes
about 30 ms on average.

This was a deliberate choice. The hard rules (identical ids, nothing off
canvas, images kept in proportion, text that really fits) must hold in
*every* output, so the engine generates layouts it can verify, never ones it
hopes are right.

The engine is still **built to take a model**. It generates several
candidate layouts, then a *ranker* picks the best one that passes every hard
rule. The ranker is interchangeable (see
[Plugging in a model](#plugging-in-a-model)). A small model, such as a
learned aesthetic scorer or a vision-language judge, can replace or
re-weight the built-in heuristic ranker without touching the layout code.
Whatever a model prefers, the hard-rule check still runs on its choice.

## Dependencies

Everything installs at build time from `requirements.txt`. Nothing is
downloaded at run time.

| Package | Why |
|---|---|
| `Pillow` | reads asset images to find their visible (non-transparent) pixels |
| `cr-renderer` (pulls in `skia-python`, `numpy`) | the official renderer; the engine also uses its fonts and its text wrapper to measure text **exactly** as it will be drawn. `numpy` (installed with it) samples image colours for the contrast check |

The Python standard library covers the rest. No GPU and no network access
are needed.

## How it works

1. **Structure (`analyze`).** The engine infers each element's role from
   geometry and layering alone, never from ids, filenames or copy:
   - background and decoration;
   - logo and hero product (found from the image's visible pixels, not its padded box);
   - text tiers: headline, body, fine print;
   - CTA and badges (pill shapes with a label);
   - text panels (an opaque card behind several texts);
   - corner accents such as tags and ribbons;
   - grouped units (`group_id` or a shape containing its text);
   - missing assets.
2. **Layout archetypes (`stack`, `split`, `row`).** The target's aspect
   ratio selects the layouts a designer would reach for:
   - a vertical stack for squares, portraits and stories (respecting story safe zones);
   - a split for landscape formats;
   - a row for strips such as 728×90, with logo left, message centre and CTA right.

   Each archetype is solved for the **largest type scale at which everything
   fits**. When space runs out, the engine gives way in this order:
   1. on display-ad sizes (short side 300 px or less): tighter margins and
      gaps when that buys at least 15% larger type, and content badges set
      as a column beside a narrow product instead of in their own row;
   2. lower minimum sizes for secondary copy, step by step.

   A solution where the logo or product would shrink out of sight is not
   accepted while a lower step keeps them recognisable.
3. **Typography.** Text is measured with the renderer's own `wrap_text` and
   glyph widths, at the integer font sizes the renderer actually draws.
   Line breaks are balanced so no line is left with a single word. CTA
   labels are centred on their cap height.
4. **Design rules in code:**
   - the CTA never ranks below badges;
   - accents keep their source corner;
   - text panels are redrawn around wherever their text lands, so copy keeps
     the backdrop it was designed on;
   - products keep their prominence;
   - one alignment axis per layout;
   - margins scale with the canvas's short side.
5. **Ranking.** Candidates are scored on:
   - collisions;
   - legibility: how close each text gets to 8 px, with text under 6 px
     counting half;
   - text contrast: each text's WCAG contrast against what is painted
     beneath it (layers composited over the white canvas, images sampled
     under the text), aiming for 4.5:1, or 3:1 for large type;
   - accent drift away from the source corner;
   - type scale;
   - use of space;
   - product prominence.
6. **Hard-rule check (`check`).** The engine checks every candidate the way
   the structural scorer does:
   - id multiset and required fields;
   - strictly on canvas, with zero tolerance: both the plain box and the
     rotated bounds;
   - image aspect;
   - integer font sizes;
   - text that fits its box;
   - unchanged content, fonts, colours and z-order.

   The best-ranked candidate that passes is returned, unless its text has
   effectively vanished: below 2:1 contrast, where the source had at least
   3:1. Such candidates are used only if nothing readable exists.
7. **Safety net.** If anything throws, or no candidate passes, the engine
   returns a conservative uniform-scale layout that always passes the hard
   rules. A task never fails outright.

No image content is ever generated, redrawn or replaced. Images are only
moved and scaled, with their aspect ratio locked.

## Plugging in a model

Set `CHAMELEON_RANKER=package.module:function` to replace the built-in
ranker:

```python
def rank(features: dict[str, float], design: dict) -> float: ...  # higher is better
```

- `features` holds `collide`, `illegible`, `drift`, `scale`, `shrink`,
  `util`, `prominence`, `legible`, and `preferred` (1.0 for the archetype that best
  suits the aspect ratio).
- `design` is the candidate's complete output JSON. A vision model can
  render it with `renderer.converter.convert.render_design_to_png` and score
  the image.

If the variable is unset, the deterministic `heuristic_rank` (a weighted sum,
`WEIGHTS` in `engine/resize.py`) is used. If the custom ranker fails to
import, or raises, the engine logs to stderr and falls back to the heuristic.
Whatever the ranker prefers, the result still has to pass the hard-rule
check.

To ship a model in the Docker image, add its package to `requirements.txt`
and download its weights during `docker build`, because scoring runs with
`--network none`. Weights for any fine-tuned model must be public.

## Validation

Dev tools live in `eval/`. They are not part of the Docker image.

```bash
python -m eval.validate 50 150 7        # submission validator (needs: pip install jsonschema)
python -m eval.stress 100 7             # random design variants x all formats
python -m eval.contact_sheet            # every input x 17 real-world formats -> output/sheets/*.png
python -m eval.contact_sheet --candidates   # each archetype's solution side by side
```

`eval.validate` checks every output against everything the scorer might
check, not only our own rules:
- the official JSON schema (`chameleon_design_schema_v0.9.json`);
- the starter's `self_check()` from `engine/main.py`;
- our hard-rule `check()`;
- strictly on canvas: plain box and rotated bounds, zero tolerance;
- only layout fields changed (`x`, `y`, `width`, `height`, `font_size`,
  `text_align`, `corner_radius`), and nothing removed;
- image proportions within 1%;
- determinism: the same input gives the same output.

It runs the samples, random variants of them, and random schema-valid
designs (fuzzing: any mix of text, images, missing assets, shapes, groups,
rotations up to 90° and optional fields) across all 17 formats.

The catalogue (`eval/catalogue.py`) has 17 formats:

- Instagram post, portrait and story;
- Facebook link and cover;
- X/Twitter post and header;
- YouTube thumbnail and Pinterest pin;
- Desktop HD (1920×1080);
- IAB display ads: 300×250, 336×280, 728×90, 970×250, 160×600, 300×600
  and 320×50.

Final validator run: **6,902 cases (samples, 100 variants and 300 random
fuzz designs × 17 formats), 0 failing.**

Stress results over 3,400 random variants × formats:

- 0 crashes and 0 hard-rule failures;
- about 2% of cases used the safe fallback, almost all 320×50 strips holding 16+ elements;
- about 25 ms per task on average on an Apple-silicon laptop.

Text that loses its contrast against its backdrop dropped from 9.3% of
stress cases to 0.6% after the contrast check. On the three sample designs
it is 0%.

**Docker, verified.** `docker build --platform linux/amd64` succeeds, and
`docker run --network none` processes all three samples. Inside the
container, all 51 sample × format cases pass every hard rule and the
starter's self-check, at about 34 ms per case under amd64 emulation.

Text is measured where the engine runs. Linux and macOS skia measure glyphs
a few pixels differently, so container JSON can differ slightly from a Mac
run, but each environment's output fits its own renderer.

## Operating limitations

We'd rather state these plainly:

- **Small display ads with heavy content.** Every input element stays
  visible, and nothing is hidden or dropped. So a dense design (the Level 3
  sample has 17 elements) has to fit 300×250, 336×280, 160×600, 728×90 or
  320×50 entirely:
  - every text stays at 6 px or above at 336×280, 728×90, 160×600,
    300×600 and 970×250. At 300×250 the corner ribbon falls to 4 px; the
    badges are 7 px and the CTA 8 px;
  - at 320×50 the engine returns the safe uniform-scale layout, which is
    letterboxed rather than re-flowed.

  Simpler designs (Levels 1–2) stay at 8 px or above everywhere except
  320×50.
- **Intentional bleed is not reproduced.** The structural check lists
  "nothing off-canvas", so every element is kept fully on canvas, including
  decorations that bled off the source. This is a deliberate safety choice.
  The `ALLOW_BLEED` switch in `engine/resize.py` can turn bleed on (tested:
  0 hard-rule failures over 1,700 stress cases), but it is off in this
  submission.
- **Contrast is estimated from mean colours.** Text over a busy photo can
  pass the mean-colour check and still be hard to read. Text contrast is
  ranked, not repaired: the engine never recolours anything.
- **No cross-size family consistency.** Each size is solved independently,
  so the logo position or text alignment can differ between sizes of the
  same design.
- **Role detection is heuristic.** Unusual structures can be misread:
  - several products;
  - a headline split across several text elements;
  - rotated text with no shape behind it;
  - an image card (not a full background) behind text.
- **Not supported:** `letter_spacing` and `max_lines` are ignored. Centred
  source text is set left-aligned in row and split layouts.
- **Ranker weights are hand-tuned** on the sample designs, not calibrated
  against the official AI evaluator.
- The engine imports a few private helpers from the vendored renderer to
  match its text measurement exactly. If the renderer changes, those
  imports must follow.

## How this meets RULES.md

| Rule | How the engine meets it |
|---|---|
| Same element `id`s, nothing added or removed | Every source element is placed exactly once; `check()` compares the id multiset. |
| Required fields present | Output elements are copies of the source elements with only layout fields updated; validated against the official schema. |
| Nothing off-canvas | `_finalize` keeps every element's plain box and rotated bounds inside the canvas (strictly, after rounding); larger-than-canvas rotated elements are shrunk about their centre. |
| Images kept in proportion | Images are scaled uniformly (aspect exact up to 0.01 px rounding). Only a cover-fit background is resized to fill the canvas, which crops rather than stretches. |
| Renders legibly | Text is measured with the renderer's own wrapper and fonts. Ranking rewards legible sizes and text contrast, and a gate rejects layouts whose text vanishes into its backdrop. |
| Content and font family fixed | `content`, `font_family`, weight, style, colour and `style_runs` are never changed; only `font_size` (integer) and the text box change. |
| No regenerated image content (disqualifying) | Pixels are only read: to find an asset's visible area and to sample colours for the contrast check. `asset_id` and `fit` are never changed, and nothing is drawn or written. |
| Models and weights | No model is used, so there are no weights to publish. |
| Latency | About 30 ms per task on average; worst case observed 0.35 s (1,071 sample and fuzz cases). |
| Hardware | CPU only, about 220 MB peak memory, no GPU, no network. |

## Repo layout

```
engine/
  resize.py         <- the engine (all of our code)
  main.py           <- organizers' orchestrator: input -> resize() -> output + render
eval/               <- our dev tools (validator, stress test, contact sheets, format catalogue); not shipped
renderer/           <- the official renderer, vendored by the organizers
brand_kit/          <- fonts and assets referenced by the designs
input/ output/      <- task JSONs in, resized JSON + PNG out
chameleon_design_schema_v0.9.json, RULES.md, FAQ.md, Dockerfile
```
