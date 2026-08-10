#!/usr/bin/env python3
"""Auditoría del dataset de penales (labels.csv ↔ img/).

Convención de frames del mismo disparo:
  penal_..._01.png          → frame base (aparece en labels.csv)
  penal_..._01_1.png        → frame extra (hereda x,y,arquero del base)
  penal_..._01_2.png        → frame extra
  ...

Columna opcional `arquero`: lado al que se tiró el arquero.
  Valores válidos: L | CL | CR | R
  Sirve como baseline humano para comparar contra el modelo.

Uso:
  python dataset/audit_dataset.py
  python dataset/audit_dataset.py --strict   # exit 1 si hay errores bloqueantes
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "labels.csv"
IMG_DIR = ROOT / "img"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}

# foo_01_3.png → base foo_01.png, frame 3
EXTRA_FRAME_RE = re.compile(r"^(.+)_([1-9]\d*)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)

ARQUERO_OK = {"L", "CL", "CR", "R"}


def parse_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or set(s) <= {"-"} or s.startswith("---"):
                continue
            if s.lower().startswith("penal,x"):
                continue
            parts = [p.strip() for p in s.split(",")]
            name = parts[0]
            xs = parts[1] if len(parts) > 1 else ""
            ys = parts[2] if len(parts) > 2 else ""
            arq = parts[3].upper() if len(parts) > 3 and parts[3] else ""
            rows.append({"line": i, "name": name, "x": xs, "y": ys, "arquero": arq})
    return rows


def shot_side_lcr(x: float) -> str:
    """Misma regla que el notebook: x<0.4 L, 0.4–0.6 C, x>0.6 R."""
    if x < 0.4:
        return "L"
    if x <= 0.6:
        return "C"
    return "R"


def shot_side_lr(x: float) -> str:
    """Binario del notebook: x<0.5 izquierda, x>=0.5 derecha."""
    return "L" if x < 0.5 else "R"


def arquero_as_lcr(arquero: str) -> str:
    """Predicción del arquero en L/C/R: CL y CR cuentan como centro."""
    return {"L": "L", "CL": "C", "CR": "C", "R": "R"}[arquero]


def arquero_as_lr(arquero: str) -> str:
    """Predicción del arquero en L/R: CL→L, CR→R."""
    return {"L": "L", "CL": "L", "CR": "R", "R": "R"}[arquero]


def keeper_covers_lcr(arquero: str, shot: str) -> bool:
    return arquero_as_lcr(arquero) == shot


def keeper_covers_lr(arquero: str, shot: str) -> bool:
    return arquero_as_lr(arquero) == shot


def looks_like_image(name: str) -> bool:
    return any(name.lower().endswith(e) for e in IMG_EXT)


def candidates(name: str) -> set[str]:
    c = {name}
    if not name.lower().endswith(tuple(IMG_EXT)):
        c.add(name + ".png")
    if name.endswith(".png"):
        c.add(name + ".png")  # foo.png.png
    return c


def resolve(name: str, images: dict[str, Path]) -> str | None:
    for c in candidates(name):
        if c in images:
            return c
    return None


def parse_extra_frame(name: str) -> tuple[str, int] | None:
    """Si name es un frame extra (..._N.ext), devuelve (base.png, N)."""
    m = EXTRA_FRAME_RE.match(name)
    if not m:
        return None
    base = f"{m.group(1)}.png"
    return base, int(m.group(2))


def zone(x: float, y: float) -> str:
    cx = "L" if x < 1 / 3 else ("C" if x <= 2 / 3 else "R")
    cy = "bajo" if y < 1 / 3 else ("medio" if y <= 2 / 3 else "alto")
    return f"{cx}-{cy}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditar dataset de penales")
    ap.add_argument("--strict", action="store_true", help="Exit 1 si hay errores")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: no existe {CSV_PATH}")
        return 1
    if not IMG_DIR.is_dir():
        print(f"ERROR: no existe {IMG_DIR}")
        return 1

    rows = parse_csv(CSV_PATH)
    labeled = [r for r in rows if r["x"] != "" and r["y"] != "" and looks_like_image(r["name"])]
    unlabeled = [r for r in rows if looks_like_image(r["name"]) and (r["x"] == "" or r["y"] == "")]
    junk = [r for r in rows if not looks_like_image(r["name"])]
    images = {p.name: p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXT}

    blocking: list[str] = []
    warnings: list[str] = []

    for r in junk:
        warnings.append(f"L{r['line']}: fila ignorada (no parece imagen) → {r['name']!r}")

    print("=== RESUMEN ===")
    print(f"filas útiles CSV: {len(rows)}")
    print(f"con x,y:         {len(labeled)}")
    print(f"sin coords:      {len(unlabeled)}")
    print(f"filas basura:    {len(junk)}")
    print(f"imágenes:        {len(images)}")
    n_arq_preview = sum(1 for r in labeled if r.get("arquero"))
    print(f"con arquero:     {n_arq_preview} / {len(labeled)}")

    matched: dict[str, str] = {}
    for r in labeled:
        hit = resolve(r["name"], images)
        if hit is None:
            blocking.append(f"L{r['line']}: labeled sin archivo → {r['name']}")
        else:
            matched[r["name"]] = hit
            if hit != r["name"]:
                warnings.append(
                    f"L{r['line']}: nombre CSV '{r['name']}' resuelve a archivo '{hit}'"
                )
            # Un frame extra no debería figurar como fila labeled propia
            if parse_extra_frame(hit) is not None:
                base, fr = parse_extra_frame(hit)  # type: ignore[misc]
                if base in matched or base in {r["name"] for r in labeled}:
                    warnings.append(
                        f"L{r['line']}: '{hit}' parece frame extra de '{base}' "
                        f"(conviene etiquetar solo el base y dejar _{fr} fuera del CSV)"
                    )

    used = set(matched.values())
    for r in unlabeled:
        hit = resolve(r["name"], images)
        if hit:
            used.add(hit)
        else:
            warnings.append(f"L{r['line']}: sin coords y sin archivo → {r['name']}")
        blocking.append(f"L{r['line']}: falta etiqueta x,y → {r['name']}")

    # Frames extra: base_N.png hereda la etiqueta del base.png si el base está labeled
    labeled_bases = set(matched.values())
    frames_by_base: dict[str, list[tuple[str, int]]] = defaultdict(list)
    linked_extras: set[str] = set()
    orphans: list[str] = []

    for n in sorted(images):
        if n in used:
            continue
        parsed = parse_extra_frame(n)
        if parsed is not None:
            base, fr = parsed
            if base in labeled_bases:
                frames_by_base[base].append((n, fr))
                linked_extras.add(n)
                continue
        orphans.append(n)

    for n in orphans:
        warnings.append(f"imagen sin fila labeled: {n}")

    for n in images:
        if n.endswith(".png.png"):
            warnings.append(f"doble extensión: {n} (renombrar a .png)")
        if re.match(r"^penal_(-?\d*)\.png$", n) or n == "penal_.png":
            blocking.append(f"nombre temporal: {n}")
        if "zimbaue" in n:
            warnings.append(f"posible typo en nombre: {n} (¿zimbabwe?)")

    no_ext = [r for r in labeled if not any(r["name"].lower().endswith(e) for e in IMG_EXT)]
    for r in no_ext:
        warnings.append(f"L{r['line']}: sin extensión en CSV → {r['name']}")

    dups = [n for n, c in Counter(r["name"] for r in labeled).items() if c > 1]
    for n in dups:
        blocking.append(f"nombre duplicado en CSV: {n}")

    xs: list[float] = []
    ys: list[float] = []
    oob: list[str] = []
    for r in labeled:
        try:
            fx, fy = float(r["x"]), float(r["y"])
        except ValueError:
            blocking.append(f"L{r['line']}: coords no numéricas → {r['name']},{r['x']},{r['y']}")
            continue
        xs.append(fx)
        ys.append(fy)
        if fx < 0 or fx > 1 or fy < 0 or fy > 1:
            oob.append(f"L{r['line']}: fuera [0,1] → {r['name']} ({fx}, {fy})")

    # --- Arquero ---
    arq_counts: Counter[str] = Counter()
    arq_missing = 0
    arq_invalid: list[str] = []
    with_arq: list[dict] = []
    for r in labeled:
        a = r.get("arquero", "")
        if not a:
            arq_missing += 1
            continue
        if a not in ARQUERO_OK:
            arq_invalid.append(f"L{r['line']}: arquero inválido '{a}' → {r['name']} (válidos: L|CL|CR|R)")
            blocking.append(arq_invalid[-1])
            continue
        arq_counts[a] += 1
        try:
            with_arq.append({"name": r["name"], "x": float(r["x"]), "y": float(r["y"]), "arquero": a})
        except ValueError:
            pass

    print("\n=== MATCHING ===")
    print(f"labeled OK:     {len(matched)}")
    print(f"labeled faltan: {sum(1 for b in blocking if 'sin archivo' in b)}")
    print(f"frames extra:   {len(linked_extras)} (heredan x,y,arquero del base)")
    print(f"huérfanas:      {len(orphans)}")
    for n in orphans:
        print(f"  {n}")

    print("\n=== FRAMES POR PENAL ===")
    n_with_extra = len(frames_by_base)
    frame_counts = [1 + len(v) for v in frames_by_base.values()]
    print(f"penales con frames extra: {n_with_extra} / {len(matched)}")
    print(f"total muestras efectivas: {len(matched) + len(linked_extras)} "
          f"(bases + extras)")
    if frame_counts:
        print(f"frames/penal (solo los que tienen extra): "
              f"min={min(frame_counts)} med={statistics.median(frame_counts):.0f} "
              f"max={max(frame_counts)}")
        # ejemplos
        samples = sorted(frames_by_base.items(), key=lambda kv: -len(kv[1]))[:5]
        print("ejemplos (base → extras):")
        for base, frs in samples:
            frs_sorted = sorted(frs, key=lambda t: t[1])
            print(f"  {base} → {[n for n, _ in frs_sorted]}")

    print("\n=== COORDENADAS ===")
    if xs:
        print(f"x: {min(xs):.3f} .. {max(xs):.3f}")
        print(f"y: {min(ys):.3f} .. {max(ys):.3f}")
    print(f"fuera [0,1] (fallos OK si es intencional): {len(oob)}")
    for line in oob:
        print(f"  {line}")

    inside = [(x, y) for x, y in zip(xs, ys) if 0 <= x <= 1 and 0 <= y <= 1]
    zc = Counter(zone(x, y) for x, y in inside)
    print("zonas (dentro [0,1]):")
    for k in sorted(zc):
        print(f"  {k}: {zc[k]}")
    center = zc.get("C-bajo", 0) + zc.get("C-medio", 0) + zc.get("C-alto", 0)
    if inside and center / len(inside) < 0.15:
        warnings.append(
            f"pocos tiros al centro ({center}/{len(inside)} = {100 * center / len(inside):.0f}%)"
        )

    print("\n=== ARQUERO ===")
    print("Valores válidos: L | CL | CR | R (lado al que se tiró el arquero)")
    n_labeled = len(labeled)
    n_with = sum(arq_counts.values())
    print(f"con arquero:  {n_with} / {n_labeled} "
          f"({100 * n_with / n_labeled:.0f}%)" if n_labeled else "con arquero: 0")
    print(f"sin arquero:  {arq_missing}")
    if arq_counts:
        print("distribución:")
        for k in ("L", "CL", "CR", "R"):
            print(f"  {k}: {arq_counts.get(k, 0)}")
    if arq_missing and n_labeled:
        warnings.append(
            f"faltan {arq_missing}/{n_labeled} etiquetas de arquero "
            f"(baseline GK incompleto hasta completarlas)"
        )

    # Baseline: ¿el arquero fue hacia el lado del tiro?
    if with_arq:
        ok_lcr = sum(
            1 for r in with_arq
            if keeper_covers_lcr(r["arquero"], shot_side_lcr(r["x"]))
        )
        ok_lr = sum(
            1 for r in with_arq
            if keeper_covers_lr(r["arquero"], shot_side_lr(r["x"]))
        )
        n = len(with_arq)
        print(f"baseline arquero vs tiro (sobre los {n} con etiqueta):")
        print(f"  acierto L/C/R: {ok_lcr}/{n} = {100 * ok_lcr / n:.1f}%  "
              f"(CL/CR → C)")
        print(f"  acierto L/R:   {ok_lr}/{n} = {100 * ok_lr / n:.1f}%  "
              f"(CL→L, CR→R)")
        print("  (compará estos % con la accuracy del modelo en el mismo subconjunto)")

    print("\n=== IMÁGENES ===")
    if Image is None:
        warnings.append("Pillow no instalado: no se auditaron tamaños (pip install pillow)")
    else:
        non_square = []
        tiny = []
        corrupt = []
        ws, hs = [], []
        for n, p in images.items():
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    ws.append(w)
                    hs.append(h)
                    if w != h:
                        non_square.append((n, w, h))
                    if min(w, h) < 128:
                        tiny.append((n, w, h))
            except Exception as e:  # noqa: BLE001
                corrupt.append((n, str(e)))
                blocking.append(f"imagen ilegible: {n} ({e})")

        if ws:
            print(f"rango: {min(ws)}×{min(hs)} .. {max(ws)}×{max(hs)}")
            print(f"mediana: {statistics.median(ws):.0f}×{statistics.median(hs):.0f}")
        print(f"no cuadradas: {len(non_square)}")
        for n, w, h in non_square:
            warnings.append(f"no cuadrada: {n} ({w}×{h})")
            print(f"  {n}: {w}×{h}")
        print(f"<128px: {len(tiny)}")
        for n, w, h in tiny:
            warnings.append(f"muy chica: {n} ({w}×{h})")
        if corrupt:
            print(f"corruptas: {corrupt}")

    print("\n=== INCEPTION V3 (notas) ===")
    print("No hace falta pre-redimensionar a mano ni forzar cuadrado en disco.")
    print("En el DataLoader: resize/crop a 299×299 + normalize ImageNet.")
    print("Frames extra (..._N.png) heredan (x,y,arquero) del base; split/CV deben agrupar por penal.")
    print("Columna arquero (L/CL/CR/R) = baseline humano para comparar contra el modelo.")

    print("\n=== BLOQUEANTES ===")
    if not blocking:
        print("(ninguno)")
    else:
        for b in blocking:
            print(f"  ✗ {b}")

    print("\n=== WARNINGS ===")
    if not warnings:
        print("(ninguno)")
    else:
        for w in warnings:
            print(f"  ! {w}")

    print("\n=== VEREDICTO ===")
    usable_bases = len(matched)
    usable_frames = usable_bases + len(linked_extras)
    print(f"penales usables (bases): {usable_bases}")
    print(f"muestras usables (bases+frames): {usable_frames}")
    if blocking:
        print("Estado: hay que corregir errores bloqueantes antes de entrenar.")
        return 1 if args.strict else 0
    print("Estado: CSV↔img OK para entrenar (revisá warnings).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
