"""Generate gryphfind/static/gryphon.svg, the faceted ("high-poly") gryphon art.
Usage: python tools/make_gryphon.py out.svgfrag  (then wrap in <svg><defs>...</defs></svg>)
Each part is a closed path (absolute M/L/C/Q/Z). We sample its outline, scatter
jittered interior points, Delaunay-triangulate, and shade each triangle with a
pseudo-3D dome normal lit from the top-left plus a little random variation."""
import colorsys, random, re, sys
import numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import Delaunay, cKDTree

random.seed(7); np.random.seed(7)
LIGHT = np.array([-0.55, -0.83])

def parse(d):
    toks = re.findall(r"[MLCQZ]|-?\d+\.?\d*", d)
    pts, i, cur, cmd = [], 0, None, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCQZ":
            cmd = t; i += 1
            if t == "Z": continue
        n = {"M": 2, "L": 2, "C": 6, "Q": 4}[cmd]
        v = list(map(float, toks[i:i + n])); i += n
        if cmd == "M":
            cur = np.array(v); pts.append(cur); cmd = "L"
        elif cmd == "L":
            p = np.array(v)
            for s in np.linspace(0, 1, 12)[1:]: pts.append(cur + (p - cur) * s)
            cur = p
        else:
            c = [cur] + [np.array(v[j:j + 2]) for j in range(0, n, 2)]
            for s in np.linspace(0, 1, 24)[1:]:
                if cmd == "C":
                    q = (1-s)**3*c[0] + 3*(1-s)**2*s*c[1] + 3*(1-s)*s**2*c[2] + s**3*c[3]
                else:
                    q = (1-s)**2*c[0] + 2*(1-s)*s*c[1] + s**2*c[2]
                pts.append(q)
            cur = c[-1]
    return np.array(pts)

def resample(poly, step):
    seg = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    L = np.hypot(*seg.T); cum = np.concatenate([[0], np.cumsum(L)])
    out = []
    for t in np.arange(0, cum[-1], step):
        k = np.searchsorted(cum, t, side="right") - 1
        f = (t - cum[k]) / L[k] if L[k] else 0
        out.append(poly[k] + seg[k] * f)
    return np.array(out)

def hexrgb(h): return [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]

def shade(base, dl, dj):
    r, g, b = hexrgb(base); h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(0.95, max(0.02, l + dl + dj))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (round(r*255), round(g*255), round(b*255))

def facet(d, color, step=15, amp=0.12, jit=0.03, outline=None):
    poly = parse(d)
    edge = resample(poly, step * 0.8)
    mp = MPath(poly)
    xmin, ymin = poly.min(0); xmax, ymax = poly.max(0)
    inner = []
    for x in np.arange(xmin, xmax, step):
        for y in np.arange(ymin, ymax, step * 0.87):
            p = np.array([x + random.uniform(-.35, .35) * step, y + random.uniform(-.35, .35) * step])
            if mp.contains_point(p, radius=-step * 0.45): inner.append(p)
    P = np.vstack([edge] + ([np.array(inner)] if inner else []))
    tri = Delaunay(P)
    tree = cKDTree(edge)
    cen_part = poly.mean(0)
    dmax = max(tree.query(P)[0].max(), 1)
    out = []
    for s in tri.simplices:
        c = P[s].mean(0)
        if not mp.contains_point(c): continue
        a, b2, c2 = P[s]
        mids = np.array([(a + b2) / 2, (b2 + c2) / 2, (a + c2) / 2])
        if not mp.contains_points(mids, radius=1.5).all(): continue
        dist = tree.query(c)[0]
        v = c - cen_part; nv = v / (np.linalg.norm(v) + 1e-6)
        rim = 1 - min(1, dist / dmax)              # 1 at edge, 0 at core
        normal = nv * rim                            # dome-ish
        lit = -(normal @ LIGHT)                      # >0 when facing light
        dl = amp * lit - amp * 0.35 * rim * (1 - max(lit, 0))
        col = shade(color, dl, random.uniform(-jit, jit))
        pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in P[s])
        out.append(f'<polygon points="{pts}" fill="{col}"/>')
    base = f'<path d="{d}" fill="{shade(color, 0, 0)}"/>'
    line = f'<path d="{d}" fill="none" stroke="{outline}" stroke-width="2.5" stroke-linejoin="round"/>' if outline else ""
    return base + "".join(out) + line

# ---- parts (back to front) ----
BLACK = "#1e1e22"
body_parts = [
    ("M232 200 C226 150 250 100 312 70 C306 96 318 104 334 100 C324 124 336 134 352 132 C340 154 350 166 364 168 C344 190 316 206 276 214 Z", "#7d0a24", 14, .10),  # far wing
    ("M292 330 C340 330 362 300 352 268 C346 250 330 252 334 266 C338 282 326 300 300 304 Z", BLACK, 9, .14),  # tail
    ("M352 268 C366 250 364 228 346 222 C352 238 346 254 336 262 Z", "#C20430", 6, .14),  # tail tuft
    ("M150 250 C150 210 190 190 235 196 C285 203 312 238 312 286 C312 330 290 350 250 352 L170 352 C150 352 140 340 142 322 C144 300 150 280 150 250 Z", BLACK, 15, .16),  # body
    ("M236 292 C276 286 304 314 298 352 L256 352 C260 336 252 326 234 326 Z", "#3a3a42", 11, .18, "#0a0a0c"),  # haunch
    ("M188 290 C196 310 198 330 196 344 C204 352 216 354 220 350 C214 330 212 308 208 288 Z", "#33333b", 7, .14, "#0a0a0c"),  # far front leg
    ("M156 294 C162 314 164 332 162 344 C170 354 186 354 188 348 C184 330 182 312 180 292 Z", "#45454e", 7, .16, "#0a0a0c"),  # near front leg
    ("M222 208 C210 150 230 86 296 46 C290 76 298 90 316 86 C304 112 314 124 330 122 C316 144 324 158 340 160 C322 182 298 200 260 212 Z", "#C20430", 12, .14),  # near wing
    ("M224 206 C220 172 232 138 262 114 C264 138 278 150 300 148 C290 170 278 188 256 204 Z", "#d9234b", 9, .12),  # wing coverts
    ("M150 262 C136 222 140 170 168 138 C186 118 216 116 232 132 C244 144 244 164 236 176 C228 190 222 206 226 226 C200 228 170 240 150 262 Z", BLACK, 10, .16),  # neck ruff
    ("M150 262 C146 236 150 208 166 192 C180 200 200 204 222 200 C220 212 222 220 226 226 C200 228 170 240 150 262 Z", "#34343a", 8, .14),  # chest feathers
]
head_parts = [
    ("M166 140 C182 120 212 116 230 128 C244 140 244 160 234 172 C222 178 196 178 176 168 C168 162 164 150 166 140 Z", "#141417", 8, .16),  # head
    ("M222 128 L248 100 L240 140 Z", "#141417", 6, .12),  # ear tuft
    ("M150 168 C152 178 148 186 140 190 C156 190 170 180 178 166 C170 168 160 170 150 168 Z", "#d9a300", 5, .10),  # lower beak
    ("M174 138 C150 134 126 142 112 162 C128 158 142 160 150 168 C160 166 172 162 180 156 Z", "#FFC72A", 6, .12),  # upper beak
]

def strokes():
    return ('<path d="M262 200 C262 164 276 132 300 106 M276 206 C286 178 302 158 324 142 M292 208 C306 194 322 184 342 176" '
            'stroke="#6f0019" stroke-width="3" fill="none" stroke-linecap="round" opacity=".7"/>'
            '<path d="M158 352 L151 362 L165 359 Z M171 352 L168 363 L178 359 Z M192 352 L185 362 L199 359 Z M205 352 L202 363 L212 359 Z" fill="#FFC72A"/>')

def eye():
    return ('<path d="M178 133 L210 137" stroke="#FFC72A" stroke-width="4" stroke-linecap="round"/>'
            '<circle cx="193" cy="147" r="9" fill="#FFC72A"/><circle cx="195" cy="147" r="4.2" fill="#000"/>'
            '<circle cx="196.6" cy="145.2" r="1.4" fill="#fff"/>'
            '<path d="M150 168 C152 178 148 186 140 190" stroke="#a87b00" stroke-width="2" fill="none"/>')

body = "\n".join(facet(d, c, p[0]*1.15, p[1], outline=(p[2] if len(p) > 2 else None)) for d, c, *p in body_parts) + strokes()
head = "\n".join(facet(d, c, s*1.1, a) for d, c, s, a in head_parts) + eye()
neck = facet(body_parts[-2][0], BLACK, 8, .16)
open(sys.argv[1], "w").write(
    f'<g id="gBody">\n{body}\n</g>\n<g id="gHead">\n{head}\n</g>\n<g id="gNeck">\n{neck}\n</g>\n')
print("triangles:", (body + head + neck).count("<polygon"))
