"""Draw the research-paper diagrams (Fig. 1 pipeline, Fig. 2 architecture) as 300-dpi PNGs
sized for one IEEE column (3.4 in). Run: python scripts/paper_figures.py"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "paper"
DPI = 300
W = int(3.4 * DPI)
FONT = "C:/Windows/Fonts/times.ttf"
BOLD = "C:/Windows/Fonts/timesbd.ttf"
INK, MUTED, LINE = (20, 20, 20), (90, 90, 90), (40, 40, 40)
FILL, FILL_ALT = (236, 244, 241), (246, 246, 246)


def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, int(size * DPI / 72))


def centered(draw, box, lines, fnt, fill=INK, gap=4):
    x0, y0, x1, y1 = box
    heights = [draw.textbbox((0, 0), s, font=f)[3] for s, f in lines]
    total = sum(heights) + gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for (s, f), h in zip(lines, heights, strict=True):
        w = draw.textlength(s, font=f)
        draw.text((x0 + (x1 - x0 - w) / 2, y), s, font=f, fill=fill)
        y += h + gap


def box(draw, xy, title, sub=None, dashed=False, fill=FILL):
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=None)
    if dashed:
        x0, y0, x1, y1 = xy
        dash = 14
        for x in range(int(x0), int(x1), dash * 2):
            draw.line([(x, y0), (min(x + dash, x1), y0)], fill=LINE, width=3)
            draw.line([(x, y1), (min(x + dash, x1), y1)], fill=LINE, width=3)
        for y in range(int(y0), int(y1), dash * 2):
            draw.line([(x0, y), (x0, min(y + dash, y1))], fill=LINE, width=3)
            draw.line([(x1, y), (x1, min(y + dash, y1))], fill=LINE, width=3)
    else:
        draw.rounded_rectangle(xy, radius=14, outline=LINE, width=3)
    lines = [(title, font(7.2, bold=True))] + ([(sub, font(6.2))] if sub else [])
    centered(draw, xy, lines, None)


def arrow(draw, p0, p1, width=4, head=16):
    draw.line([p0, p1], fill=LINE, width=width)
    (x0, y0), (x1, y1) = p0, p1
    if abs(x1 - x0) > abs(y1 - y0):
        s = 1 if x1 > x0 else -1
        pts = [(x1, y1), (x1 - s * head, y1 - head * 0.6), (x1 - s * head, y1 + head * 0.6)]
    else:
        s = 1 if y1 > y0 else -1
        pts = [(x1, y1), (x1 - head * 0.6, y1 - s * head), (x1 + head * 0.6, y1 - s * head)]
    draw.polygon(pts, fill=LINE)


def pipeline() -> Path:
    steps = [
        ("Data acquisition", "CPCB/KSPCB via OpenAQ, ERA5", False),
        ("Validation & cleaning", "units, flags, dedup, gaps", False),
        ("Hyperlocal matching", "haversine + IDW, freshness", False),
        ("Feature engineering", "35 past-only features", False),
        ("24 h AQ forecast", "LightGBM, 24 direct models", False),
        ("Exposure estimate", "ambient × microenvironment", False),
        ("Respiratory risk*", "asthma / COPD heads", True),
        ("Calibration & levels*", "Low … Severe policy", True),
        ("Alerts & REST API", "FastAPI, JWT, consent", False),
        ("Android app*", "Flutter client", True),
    ]
    cols, rows = 2, 5
    pad, gx, gy = 30, 70, 46
    bw = (W - 2 * pad - gx) / cols
    bh = 150
    H = int(pad * 2 + rows * bh + (rows - 1) * gy + 60)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    pos = []
    for i in range(len(steps)):
        r = i // cols
        c = i % cols if r % 2 == 0 else cols - 1 - i % cols  # snake order
        x0 = pad + c * (bw + gx)
        y0 = pad + r * (bh + gy)
        pos.append((x0, y0, x0 + bw, y0 + bh))
    for (t, s, dashed), xy in zip(steps, pos, strict=True):
        box(d, xy, t, s, dashed=dashed, fill=FILL_ALT if dashed else FILL)
    for a, b in zip(pos, pos[1:], strict=False):
        if abs(a[1] - b[1]) < 1:  # same row
            y = (a[1] + a[3]) / 2
            if b[0] > a[0]:
                arrow(d, (a[2] + 4, y), (b[0] - 6, y))
            else:
                arrow(d, (a[0] - 4, y), (b[2] + 6, y))
        else:
            x = (a[0] + a[2]) / 2
            arrow(d, (x, a[3] + 4), (x, b[1] - 6))
    note = "* in progress (Phases 4–6). Solid boxes are implemented and evaluated."
    d.text((pad, H - 50), note, font=font(6), fill=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig1_pipeline.png"
    img.save(path, dpi=(DPI, DPI))
    return path


def architecture() -> Path:
    pad = 30
    H = 1180
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    inner = W - 2 * pad
    ext_w = 250
    main_x0, main_x1 = pad, W - pad - ext_w - 50

    layers = [
        ("Clients", "Flutter Android app*  ·  OpenAPI /docs", FILL_ALT),
        ("FastAPI  /api/v1", "JWT auth · ownership checks · consent · error contract", FILL),
        ("Services", "air · forecast · exposure · ingestion · patients", FILL),
        ("Domain & ML", "spatial IDW · cleaning · features · LightGBM artifacts", FILL),
        ("PostgreSQL 16", "observations · weather · stations · patients · audit", FILL),
    ]
    lh, gap = 160, 62
    ys = []
    for i, (t, s, f) in enumerate(layers):
        y0 = pad + i * (lh + gap)
        xy = (main_x0, y0, main_x1, y0 + lh)
        ys.append(xy)
        box(d, xy, t, s, dashed=(i == 0), fill=f)
    for a, b in zip(ys, ys[1:], strict=False):
        x = (a[0] + a[2]) / 2
        arrow(d, (x - 40, a[3] + 4), (x - 40, b[1] - 6))
        arrow(d, (x + 40, b[1] - 4), (x + 40, a[3] + 6))

    ex0, ex1 = main_x1 + 50, W - pad
    ext = (ex0, ys[0][1], ex1, ys[2][3])
    d.rounded_rectangle(ext, radius=14, fill=FILL_ALT, outline=LINE, width=3)
    centered(d, (ex0, ext[1] + 20, ex1, ext[1] + 110), [("External", font(7.2, bold=True)), ("sources", font(7.2, bold=True))], None)
    centered(
        d,
        (ex0, ext[1] + 130, ex1, ext[3] - 20),
        [(s, font(6.2)) for s in ("OpenAQ v3", "(CPCB/KSPCB)", "", "Open-Meteo", "ERA5 + forecast", "", "data.gov.in*")],
        None,
    )
    cli = (ex0, ys[3][1], ex1, ys[3][3])
    box(d, cli, "Ingestion CLI", "validate · dedup", fill=FILL)
    cx = (ex0 + ex1) / 2
    arrow(d, (cx, ext[3] + 4), (cx, cli[1] - 6))
    arrow(d, (cx, cli[3] + 4), (cx, ys[4][1] + 20))
    d.line([(cx, ys[4][1] + 20), (main_x1 + 6, ys[4][1] + 20)], fill=LINE, width=4)
    arrow(d, (cx - 1, ys[4][1] + 20), (main_x1 + 6, ys[4][1] + 20))
    d.text((pad, H - 50), "* planned / disabled until verified. Arrows show request and data flow.", font=font(6), fill=MUTED)
    path = OUT / "fig2_architecture.png"
    img.save(path, dpi=(DPI, DPI))
    _ = inner
    return path


if __name__ == "__main__":
    for p in (pipeline(), architecture()):
        print(p)
