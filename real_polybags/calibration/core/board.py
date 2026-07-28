"""
ChArUco board definition and print-ready output — OVGU AMS calibration tool.

A calibration board is only useful if its *physical* geometry is known exactly.
Two things routinely destroy that, and this module is built to prevent both:

1. **Scaled printing.** Printing "fit to page" silently shrinks the board by a
   few percent. Every distance derived from it is then wrong by that factor,
   and nothing downstream can detect it — the calibration looks fine and the
   metric results are quietly incorrect. Each page therefore carries a printed
   100 mm reference bar to check with a ruler before use.
2. **Losing the parameters.** A printed board whose square size or dictionary
   is unknown afterwards is scrap. Every page prints its own full spec in the
   footer, so the artefact is self-describing.

Two boards are defined because this rig cannot be served by one. `basler_1` is
an extreme close-up (the belt fills its frame), while `lucid`, `basler_2` and
the RealSense see the whole belt width. A board large enough to be detected
reliably by the wide cameras does not fit inside `basler_1`'s field of view;
one small enough for `basler_1` is too coarse for the others.

The two boards use **different ArUco dictionaries** so they can never be
confused with each other, even if both appear in the same shot.

Usage:
    python3 -m calibration.core.board            # write both PDFs + PNGs
    python3 core/board.py --preset small --dpi 600
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MM_PER_INCH = 25.4

# Paper sizes in mm (portrait)
PAPER = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
}

# Dictionary name -> OpenCV constant. Kept as strings in the spec so a saved
# calibration remains readable without importing OpenCV to interpret it.
DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_250": aruco.DICT_6X6_250,
}


@dataclass(frozen=True)
class BoardSpec:
    """Physical description of one printed ChArUco board."""
    name: str
    squares_x: int          # columns
    squares_y: int          # rows
    square_mm: float
    marker_mm: float
    dictionary: str
    paper: str

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def n_markers(self) -> int:
        # ChArUco places a marker in every other (white) square.
        return (self.squares_x * self.squares_y) // 2

    @property
    def n_corners(self) -> int:
        # Interior chessboard corners — these are what calibration actually uses.
        return (self.squares_x - 1) * (self.squares_y - 1)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the spec is sound."""
        problems = []
        if self.dictionary not in DICTS:
            problems.append(f"unknown dictionary {self.dictionary!r}")
        else:
            avail = aruco.getPredefinedDictionary(DICTS[self.dictionary]).bytesList.shape[0]
            if self.n_markers > avail:
                problems.append(
                    f"needs {self.n_markers} markers but {self.dictionary} only has {avail}")
        if self.marker_mm >= self.square_mm:
            problems.append("marker_mm must be smaller than square_mm")
        if self.marker_mm < 0.5 * self.square_mm:
            problems.append("marker_mm below 0.5x square_mm — markers get hard to detect")
        if self.paper not in PAPER:
            problems.append(f"unknown paper size {self.paper!r}")
        else:
            pw, _ = PAPER[self.paper]
            if self.width_mm > pw - 10:
                problems.append(
                    f"board {self.width_mm:.0f}mm too wide for {self.paper} ({pw:.0f}mm)")
            fits, spare = layout_fits(self)
            if not fits:
                problems.append(
                    f"board {self.height_mm:.0f}mm + {FOOTER_MM:g}mm footer overflows "
                    f"{self.paper} by {-spare:.1f}mm")
        return problems

    def board(self) -> aruco.CharucoBoard:
        d = aruco.getPredefinedDictionary(DICTS[self.dictionary])
        # OpenCV takes metres/consistent units; mm throughout keeps the printed
        # spec and the solver in the same units with no hidden conversion.
        return aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_mm, self.marker_mm, d)


# ── Presets ──────────────────────────────────────────────────────────────────
# Sized so each board fills a good fraction of its target camera's view while
# leaving printable margins. Different dictionaries keep them distinguishable.
PRESETS = {
    # squares_y chosen so board + footer fits the sheet (see layout_fits);
    # square_mm kept to round numbers so a ruler check is unambiguous.
    "small": BoardSpec(
        name="AMS-small",
        squares_x=7, squares_y=8,
        square_mm=25.0, marker_mm=18.0,
        dictionary="DICT_4X4_50",
        paper="A4",
    ),
    "large": BoardSpec(
        name="AMS-large",
        squares_x=7, squares_y=9,
        square_mm=36.0, marker_mm=26.0,
        dictionary="DICT_5X5_100",
        paper="A3",
    ),
}


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / MM_PER_INCH * dpi))


def render_board(spec: BoardSpec, dpi: int) -> Image.Image:
    """Render just the board pattern, at exact physical scale for `dpi`."""
    w_px = mm_to_px(spec.width_mm, dpi)
    h_px = mm_to_px(spec.height_mm, dpi)
    # marginSize=0: the page layout adds margins, so the pattern itself stays
    # exactly squares_x * square_mm wide. Any margin here would silently change
    # the physical size of each square.
    img = spec.board().generateImage((w_px, h_px), marginSize=0, borderBits=1)
    return Image.fromarray(img).convert("L")


def _font(px: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default()


# Vertical space the footer needs, in mm. Kept explicit so the layout can be
# checked against the paper size instead of discovered by looking at the output
# — the first version silently ran the ruler and the print instructions off the
# bottom of the page, which is exactly the kind of defect that survives to the
# printer.
FOOTER_MM = 52.0
TOP_MARGIN_MM = 8.0
BOTTOM_MARGIN_MM = 8.0


def layout_fits(spec: BoardSpec) -> tuple[bool, float]:
    """(fits, spare_mm) for this spec's board + footer on its paper."""
    _, ph_mm = PAPER[spec.paper]
    needed = TOP_MARGIN_MM + spec.height_mm + FOOTER_MM + BOTTOM_MARGIN_MM
    return needed <= ph_mm, ph_mm - needed


def render_page(spec: BoardSpec, dpi: int) -> Image.Image:
    """Board centred on its paper size, with spec footer and a 100 mm ruler."""
    fits, spare = layout_fits(spec)
    if not fits:
        raise ValueError(
            f"{spec.name}: board {spec.width_mm:g}x{spec.height_mm:g}mm plus "
            f"{FOOTER_MM:g}mm footer does not fit {spec.paper} "
            f"(short by {-spare:.1f}mm). Reduce squares_y or square_mm.")

    pw_mm, ph_mm = PAPER[spec.paper]
    page = Image.new("L", (mm_to_px(pw_mm, dpi), mm_to_px(ph_mm, dpi)), 255)
    board_img = render_board(spec, dpi)

    x = (page.width - board_img.width) // 2
    y = mm_to_px(TOP_MARGIN_MM, dpi)
    page.paste(board_img, (x, y))

    draw = ImageDraw.Draw(page)
    f_title = _font(mm_to_px(4.5, dpi))
    f_body = _font(mm_to_px(3.2, dpi))

    ty = y + board_img.height + mm_to_px(8.0, dpi)
    draw.text((x, ty), f"OVGU AMS — calibration board  [{spec.name}]", font=f_title, fill=0)
    ty += mm_to_px(7.0, dpi)

    for line in (
        f"{spec.squares_x} x {spec.squares_y} squares   "
        f"square {spec.square_mm:g} mm   marker {spec.marker_mm:g} mm",
        f"{spec.dictionary}   {spec.n_corners} corners   {spec.n_markers} markers   "
        f"board {spec.width_mm:g} x {spec.height_mm:g} mm",
    ):
        draw.text((x, ty), line, font=f_body, fill=0)
        ty += mm_to_px(5.0, dpi)

    # ── 100 mm verification ruler ────────────────────────────────────────────
    # The single most valuable thing on the page. If this does not measure
    # exactly 100 mm on the print, the page was scaled and every metric result
    # derived from this board would be wrong by that factor.
    ty += mm_to_px(3.0, dpi)
    bar_len = mm_to_px(100.0, dpi)
    bar_h = mm_to_px(3.0, dpi)
    draw.rectangle([x, ty, x + bar_len, ty + bar_h], outline=0, width=max(1, dpi // 300))
    for i in range(11):                      # 10 mm ticks
        tx = x + mm_to_px(10.0 * i, dpi)
        tick = bar_h * (2 if i % 5 == 0 else 1)
        draw.line([tx, ty + bar_h, tx, ty + bar_h + tick], fill=0, width=max(1, dpi // 300))
    draw.text((x, ty + bar_h * 4), "<- this bar must measure exactly 100 mm ->",
              font=f_body, fill=0)
    ty += bar_h * 4 + mm_to_px(6.0, dpi)

    draw.text((x, ty), "PRINT AT 100% / ACTUAL SIZE — do not use 'fit to page'.",
              font=f_body, fill=0)
    draw.text((x, ty + mm_to_px(5.0, dpi)),
              "Matte paper. Mount flat and rigid (foamboard). Keep this footer.",
              font=f_body, fill=0)
    return page


def write_outputs(spec: BoardSpec, out_dir: Path, dpi: int = 300) -> dict:
    problems = spec.validate()
    if problems:
        raise ValueError(f"invalid board spec {spec.name}: " + "; ".join(problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(spec, dpi)

    pdf = out_dir / f"charuco_{spec.name}_{spec.paper}.pdf"
    png = out_dir / f"charuco_{spec.name}_{spec.paper}.png"
    # PIL derives the PDF page size from the image DPI, so this lands at exact
    # physical scale rather than "whatever fits".
    page.convert("RGB").save(pdf, "PDF", resolution=float(dpi))
    page.save(png, dpi=(dpi, dpi))

    spec_json = out_dir / f"charuco_{spec.name}.json"
    import json
    spec_json.write_text(json.dumps(asdict(spec), indent=2) + "\n")

    return {"pdf": pdf, "png": png, "spec": spec_json,
            "page_px": (page.width, page.height), "dpi": dpi}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS) + ["both"], default="both")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Render DPI (default 300). Higher is sharper on paper; "
                         "physical size is unchanged.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "boards"))
    args = ap.parse_args()

    names = list(PRESETS) if args.preset == "both" else [args.preset]
    out_dir = Path(args.out)
    for n in names:
        spec = PRESETS[n]
        res = write_outputs(spec, out_dir, args.dpi)
        print(f"[{spec.name}] {spec.squares_x}x{spec.squares_y} squares, "
              f"{spec.square_mm:g}mm square, {spec.dictionary}")
        print(f"    board {spec.width_mm:g} x {spec.height_mm:g} mm on {spec.paper}, "
              f"{spec.n_corners} corners, {spec.n_markers} markers")
        print(f"    -> {res['pdf'].name}  ({res['page_px'][0]}x{res['page_px'][1]} px @ {res['dpi']} dpi)")
        print(f"    -> {res['png'].name}")
    print(f"\nWritten to {out_dir}")
    print("Send the PDF(s) to whoever prints them. Print at 100% / actual size,")
    print("then check the 100 mm bar with a ruler before using the board.")


if __name__ == "__main__":
    main()
