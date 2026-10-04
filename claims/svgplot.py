"""Dependency-free SVG line/bar charts so reports render anywhere (GitHub, browser, IDE)."""
from __future__ import annotations

from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

PALETTE = ["#2563eb", "#dc2626", "#d97706", "#059669", "#7c3aed", "#db2777", "#0891b2"]


def line_chart(series: Dict[str, Sequence[Tuple[float, float]]], title: str, xlabel: str, ylabel: str,
               vlines: Optional[List[Tuple[float, str]]] = None, hlines: Optional[List[Tuple[float, str]]] = None,
               ymax: Optional[float] = None, xmin: Optional[float] = None, xmax: Optional[float] = None,
               width: int = 720, height: int = 420, diagonal: bool = False) -> str:
    L, R, T, B = 64, 170, 40, 52
    pw, ph = width - L - R, height - T - B
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts]
    x0 = min(xs) if xmin is None else xmin
    x1 = max(xs) if xmax is None else xmax
    y1 = ymax if ymax is not None else (max(ys) * 1.08 if ys and max(ys) > 0 else 1.0)
    y0 = 0.0
    sx = lambda x: L + (x - x0) / ((x1 - x0) or 1) * pw  # noqa: E731
    sy = lambda y: T + ph - (y - y0) / ((y1 - y0) or 1) * ph  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="bold" fill="#111">{escape(title)}</text>']
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{sy(yv) + 4:.1f}" text-anchor="end" fill="#555">{_fmt(yv)}</text>')
        xv = x0 + (x1 - x0) * i / 5
        out.append(f'<text x="{sx(xv):.1f}" y="{T + ph + 18}" text-anchor="middle" fill="#555">{_fmt(xv)}</text>')
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}" stroke="#333"/>')
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="#333"/>')
    out.append(f'<text x="{L + pw / 2}" y="{height - 10}" text-anchor="middle" fill="#333">{escape(xlabel)}</text>')
    out.append(f'<text transform="translate(16,{T + ph / 2}) rotate(-90)" text-anchor="middle" fill="#333">{escape(ylabel)}</text>')
    if diagonal:
        out.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#9ca3af" stroke-dasharray="4 4"/>')
    for v, lab in vlines or []:
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{T + ph}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5 3"/>')
        out.append(f'<text x="{sx(v) + 4:.1f}" y="{T + 12}" fill="#ef4444" font-weight="bold">{escape(lab)}</text>')
    for v, lab in hlines or []:
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#6b7280" stroke-dasharray="2 3"/>')
        out.append(f'<text x="{L + pw - 4}" y="{sy(v) - 4:.1f}" text-anchor="end" fill="#6b7280">{escape(lab)}</text>')
    for i, (name, pts) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(min(y, y1)):.1f}" for j, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        if len(pts) <= 30:
            out += [f'<circle cx="{sx(x):.1f}" cy="{sy(min(y, y1)):.1f}" r="3" fill="{col}"/>' for x, y in pts]
        ly = T + 10 + i * 20
        out.append(f'<rect x="{L + pw + 14}" y="{ly - 9}" width="14" height="4" fill="{col}"/>')
        out.append(f'<text x="{L + pw + 34}" y="{ly - 3}" fill="#222">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def bar_chart(groups: Dict[str, Dict[str, float]], title: str, ylabel: str, width: int = 760, height: int = 420) -> str:
    """groups: {category: {series: value}} -> grouped bars."""
    L, R, T, B = 64, 170, 40, 70
    pw, ph = width - L - R, height - T - B
    cats = list(groups)
    series = list(dict.fromkeys(s for g in groups.values() for s in g))
    vmax = max([v for g in groups.values() for v in g.values()] + [1e-9]) * 1.1
    gw = pw / max(1, len(cats))
    bw = gw * 0.8 / max(1, len(series))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif" font-size="12">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{L}" y="24" font-size="15" font-weight="bold" fill="#111">{escape(title)}</text>',
           f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="#333"/>']
    for i in range(6):
        v = vmax * i / 5
        y = T + ph - v / vmax * ph
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{_fmt(v)}</text>')
    for ci, c in enumerate(cats):
        gx = L + ci * gw + gw * 0.1
        for si, s in enumerate(series):
            v = groups[c].get(s, 0.0)
            h = v / vmax * ph
            out.append(f'<rect x="{gx + si * bw:.1f}" y="{T + ph - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" fill="{PALETTE[si % len(PALETTE)]}"/>')
        out.append(f'<text x="{L + ci * gw + gw / 2:.1f}" y="{T + ph + 16}" text-anchor="middle" fill="#333">{escape(c)}</text>')
    for si, s in enumerate(series):
        ly = T + 10 + si * 20
        out.append(f'<rect x="{L + pw + 14}" y="{ly - 10}" width="12" height="12" fill="{PALETTE[si % len(PALETTE)]}"/>')
        out.append(f'<text x="{L + pw + 32}" y="{ly}" fill="#222">{escape(s)}</text>')
    out.append(f'<text transform="translate(16,{T + ph / 2}) rotate(-90)" text-anchor="middle" fill="#333">{escape(ylabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"
