"""Generate a single, simple HTML report from the JSON files produced by the
other scripts (run_audit.py, robustness_test.py, code_quality_check.py) — one
page, one section per audited dimension, intuitive at a glance. Opens in any
browser, no internet connection needed.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --audit audit_report.json --robustness robustness_report.json --code code_quality_report.json --out report.html
"""
import argparse
import json
from pathlib import Path


def _load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metric_tile(label, value):
    display = f"{value:.1%}" if isinstance(value, float) else str(value)
    return f'<div class="tile"><div class="tile-value">{display}</div><div class="tile-label">{label}</div></div>'


def _confusion_matrix_table(matrix, label_names):
    max_value = max((cell for row in matrix for cell in row), default=1) or 1
    header = "".join(f"<th>{name}</th>" for name in label_names) if label_names else ""
    rows = []
    for i, row in enumerate(matrix):
        row_label = label_names[i] if label_names and i < len(label_names) else str(i)
        cells = []
        for value in row:
            intensity = value / max_value
            cells.append(f'<td style="background: rgba(47,111,94,{intensity:.2f})">{value}</td>')
        rows.append(f"<tr><th>{row_label}</th>{''.join(cells)}</tr>")
    return f'<table class="confusion"><tr><th></th>{header}</tr>{"".join(rows)}</table>'


def _pending_section(title, command):
    return f"""
    <section class="pending">
      <h2>{title}</h2>
      <p class="meta">Todavía no se corrió. Generalo con:</p>
      <pre>{command}</pre>
    </section>
    """


def _effectiveness_section(report):
    if report is None:
        return _pending_section("1. Efectividad", "python scripts/run_audit.py")

    tiles = "".join([
        _metric_tile("Accuracy", report["accuracy"]),
        _metric_tile("Precisión", report["precision_weighted"]),
        _metric_tile("Recall", report["recall_weighted"]),
        _metric_tile("F1", report["f1_weighted"]),
    ])
    matrix = _confusion_matrix_table(report["confusion_matrix"], report.get("confusion_matrix_labels"))
    return f"""
    <section>
      <h2>1. Efectividad</h2>
      <p class="explain">Qué tan seguido acierta el modelo. <strong>Accuracy</strong> es el porcentaje total de aciertos. <strong>Precisión</strong> mide, de lo que el modelo marcó como una clase, cuánto era realmente esa clase. <strong>Recall</strong> mide, de todos los casos reales de una clase, cuántos detectó. <strong>F1</strong> combina ambas en un solo número.</p>
      <p class="meta">{report['model']} sobre {report['dataset']} &middot; {report['images_evaluated']} imágenes</p>
      <div class="tiles">{tiles}</div>
      <h3>Matriz de confusión</h3>
      <p class="explain">Muestra exactamente qué clase confunde con cuál — cada fila es la clase real, cada columna la que predijo el modelo. Los números fuera de la diagonal son los errores.</p>
      {matrix}
    </section>
    """


def _robustness_section(report):
    if report is None:
        return _pending_section("2. Robustez", "python scripts/robustness_test.py")

    baseline = report["baseline"]["accuracy"]
    perturbed = report["perturbed"]["accuracy"]
    drop = report["accuracy_drop"]
    bar = f"""
    <div class="bar-row"><span>Original</span><div class="bar"><div class="bar-fill" style="width:{baseline * 100:.0f}%"></div></div><span>{baseline:.1%}</span></div>
    <div class="bar-row"><span>Con ruido</span><div class="bar"><div class="bar-fill warn" style="width:{perturbed * 100:.0f}%"></div></div><span>{perturbed:.1%}</span></div>
    """
    verdict = "se mantiene bien" if drop < 0.05 else ("baja de forma notable" if drop < 0.15 else "es frágil ante el ruido")
    return f"""
    <section>
      <h2>2. Robustez</h2>
      <p class="explain">Compara cómo responde el modelo con las fotos originales contra las mismas fotos con ruido (borrosas, giradas, con mal brillo) — simula condiciones reales imperfectas, no solo fotos "de manual".</p>
      <p class="meta">{report['model']} sobre {report['dataset']} &middot; ruido de desenfoque + rotación + brillo/contraste</p>
      {bar}
      <p class="verdict">Caída de accuracy: <strong>{drop:.1%}</strong> &mdash; el modelo {verdict}.</p>
    </section>
    """


def _code_quality_section(report):
    if report is None:
        return _pending_section(
            "3. Implementación / código",
            "python scripts/code_quality_check.py --path /ruta/al/codigo/del/modelo",
        )

    lint = report["lint"]
    lint_ok = lint["exit_code"] == 0
    dep = report["dependency_audit"]
    dep_skipped = "skipped" in dep
    dep_ok = dep_skipped or dep.get("exit_code") == 0
    return f"""
    <section>
      <h2>3. Implementación / código</h2>
      <p class="explain">Esto no mide cómo responde el modelo, sino si el código que lo entrena o lo sirve está bien escrito y sin dependencias con fallas de seguridad conocidas.</p>
      <p class="meta">{report['path']}</p>
      <div class="tiles">
        <div class="tile"><div class="tile-value {'ok' if lint_ok else 'bad'}">{'Limpio' if lint_ok else 'Con problemas'}</div><div class="tile-label">Lint (ruff)</div></div>
        <div class="tile"><div class="tile-value {'ok' if dep_ok else 'bad'}">{'Omitido' if dep_skipped else ('Limpio' if dep_ok else 'Vulnerabilidades encontradas')}</div><div class="tile-label">Dependencias (pip-audit)</div></div>
      </div>
      <details><summary>Ver salida completa del lint</summary><pre>{lint['output'] or '(sin salida)'}</pre></details>
    </section>
    """


PAGE_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de Auditoría del Modelo</title>
<style>
  :root {{
    --bg: #f6f7f5; --surface: #ffffff; --text: #1a2420; --muted: #5c6864;
    --border: #dde3de; --accent: #2f6f5e; --accent-soft: #e3efea; --bad: #b3261e;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#101512; --surface:#161d1a; --text:#e6ece8; --muted:#8fa098; --border:#263230; --accent:#59b79a; --accent-soft:#1b2b26; --bad:#ff8a80; }}
  }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; margin: 0; padding: 24px 16px 64px; line-height: 1.5; }}
  .page {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); margin-top: 0; margin-bottom: 32px; font-size: 14px; }}
  section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; margin-bottom: 20px; }}
  section.pending {{ opacity: 0.7; border-style: dashed; }}
  h2 {{ margin-top: 0; font-size: 17px; }}
  .meta {{ color: var(--muted); font-size: 13px; margin-top: -6px; }}
  .tiles {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
  .tile {{ flex: 1; min-width: 90px; background: var(--accent-soft); border-radius: 8px; padding: 12px; text-align: center; }}
  .tile-value {{ font-size: 20px; font-weight: 600; color: var(--accent); }}
  .tile-value.bad {{ color: var(--bad); }}
  .tile-label {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  table.confusion {{ border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
  table.confusion th, table.confusion td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: center; color: var(--text); }}
  .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 13px; }}
  .bar-row span:first-child {{ width: 80px; color: var(--muted); }}
  .bar {{ flex: 1; background: var(--accent-soft); border-radius: 4px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--accent); }}
  .bar-fill.warn {{ background: #c98a1f; }}
  .verdict {{ margin-top: 10px; }}
  .explain {{ font-size: 13px; color: var(--muted); margin: 6px 0 14px; }}
  pre {{ background: var(--accent-soft); border-radius: 6px; padding: 10px 12px; overflow-x: auto; font-size: 12px; }}
  details summary {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
  <div class="page">
    <h1>Reporte de Auditoría del Modelo</h1>
    <p class="subtitle">Generado por image-model-audit-kit &mdash; abrí este archivo en cualquier navegador, no hace falta internet.</p>
    {sections}
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="audit_report.json")
    parser.add_argument("--robustness", default="robustness_report.json")
    parser.add_argument("--code", default="code_quality_report.json")
    parser.add_argument("--out", default="report.html")
    args = parser.parse_args()

    sections = "".join([
        _effectiveness_section(_load(args.audit)),
        _robustness_section(_load(args.robustness)),
        _code_quality_section(_load(args.code)),
    ])

    html = PAGE_TEMPLATE.format(sections=sections)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Reporte generado en {args.out} -- abrilo en cualquier navegador.")


if __name__ == "__main__":
    main()
