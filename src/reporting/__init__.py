"""HTML report generation for SwimVision using simple string-replacement templates."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


CSS = """<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1923;color:#e0e8f0;padding:2rem}
.header{border-bottom:2px solid #1e90ff;padding-bottom:1rem;margin-bottom:2rem}
.header h1{font-size:2rem;color:#1e90ff}
.meta{color:#8899aa;font-size:.9rem;margin-top:.5rem}
.severity{display:inline-block;padding:.25rem .75rem;border-radius:1rem;font-weight:600;font-size:.85rem}
.severity-OPTIMAL{background:#1a3a1a;color:#4caf50}
.severity-MINOR{background:#3a3510;color:#ffc107}
.severity-SIGNIFICANT{background:#3a2010;color:#ff9800}
.severity-CRITICAL{background:#3a1010;color:#f44336}
.section{background:#1a2634;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem}
.section h2{font-size:1.25rem;color:#64b5f6;margin-bottom:1rem;border-bottom:1px solid #2a3a4a;padding-bottom:.5rem}
table{width:100%;border-collapse:collapse}
th,td{padding:.5rem .75rem;text-align:left;border-bottom:1px solid #2a3a4a}
th{color:#8899aa;font-size:.8rem;text-transform:uppercase}
.flag{padding:.15rem .5rem;border-radius:4px;font-size:.75rem;font-weight:600}
.flag-OPTIMAL{background:#1a3a1a;color:#4caf50}
.flag-MINOR{background:#3a3510;color:#ffc107}
.flag-SIGNIFICANT{background:#3a2010;color:#ff9800}
.flag-CRITICAL{background:#3a1010;color:#f44336}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
.card{background:#243447;border-radius:6px;padding:1rem;text-align:center}
.card .label{color:#8899aa;font-size:.8rem;text-transform:uppercase}
.card .value{font-size:1.5rem;font-weight:700;color:#e0e8f0;margin-top:.25rem}
.card .sub{font-size:.75rem;color:#8899aa}
.rec{background:#1a2a3a;border-left:3px solid #1e90ff;padding:.75rem 1rem;margin-bottom:.5rem;border-radius:4px}
.footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #2a3a4a;color:#556677;font-size:.8rem;text-align:center}
.risk-LOW{color:#4caf50}.risk-MEDIUM{color:#ffc107}.risk-HIGH{color:#ff9800}.risk-CRITICAL{color:#f44336;font-weight:700}
.score{font-size:3rem;font-weight:800;color:#4fc3f7;text-align:center;margin:1rem 0}
.bar{width:100%;height:8px;background:#1a2a4a;border-radius:4px;overflow:hidden;margin:.5rem 0}
.bar-fill{height:100%;border-radius:4px}
.tip{background:#152238;border-left:3px solid #4fc3f7;padding:.75rem 1rem;margin-bottom:.5rem;border-radius:4px;font-size:.9rem}
.tip.good{border-left-color:#4caf50}
.tip.focus{border-left-color:#ff9800}
pre{background:#161b22;padding:1rem;border-radius:6px;overflow-x:auto;font-size:.8rem;line-height:1.5}
</style>"""


def _render_cards(cards: List[Dict[str, str]]) -> str:
    """Render metric cards as HTML."""
    if not cards:
        return ""
    rows = []
    for c in cards:
        sub = f'<div class="sub">{c["sub"]}</div>' if c.get("sub") else ""
        rows.append(f'<div class="card"><div class="label">{c["label"]}</div><div class="value">{c["value"]}</div>{sub}</div>')
    return "\n".join(rows)


def _render_table(rows: List[Dict[str, Any]], columns: List[str], research: bool = False) -> str:
    """Render a table from row dicts."""
    if not rows:
        return '<tr><td colspan="{}">No data</td></tr>'.format(len(columns))
    html_rows = []
    for r in rows:
        cells = []
        for col in columns:
            val = r.get(col, "")
            if col == "flag":
                cells.append(f'<td><span class="flag flag-{val}">{val}</span></td>')
            elif col in ("measured", "deviation") and val is not None:
                cells.append(f'<td>{float(val):.1f}</td>')
            elif col in ("optimal_min", "optimal_max") and val is not None:
                cells.append(f'<td>{float(val):.1f}</td>')
            else:
                cells.append(f"<td>{val}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(html_rows)


def _render_list(items: List[str]) -> str:
    """Render recommendations/stips as HTML divs."""
    if not items:
        return ""
    return "\n".join(f'<div class="rec">{item}</div>' for item in items)


def _render_tips(items: List[str], cls: str = "") -> str:
    """Render athlete feedback tips as HTML."""
    if not items:
        return ""
    css = f" {cls}" if cls else ""
    return "\n".join(f'<div class="tip{css}">{item}</div>' for item in items)


# ── Coach Report ──────────────────────────────────────────────

def generate_coach_report(
    clip_id: str,
    deviations: Dict[str, Any],
    swimmer_id: str = "unknown",
    reaction_time_ms: Optional[float] = None,
    risk_level: Optional[str] = None,
    predictions: Optional[Dict[str, Any]] = None,
    entry_analysis: Optional[Dict[str, Any]] = None,
    glide_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a coach-oriented HTML report."""
    overall_severity = deviations.get("overall_severity", "OPTIMAL")

    # Key metric cards
    cards: List[Dict[str, str]] = []
    if reaction_time_ms is not None:
        cards.append({"label": "Reaction Time", "value": f"{reaction_time_ms:.0f}ms"})
    cards.append({"label": "Overall", "value": overall_severity})
    flagged = [
        r for phase in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle")
        for r in deviations.get(phase, []) if isinstance(r, dict) and r.get("flag") != "OPTIMAL"
    ]
    cards.append({"label": "Issues Found", "value": str(len(flagged))})
    if predictions:
        cards.append({"label": "Est. 50m Time", "value": f"{predictions.get('estimated_50m_time_sec', '?'):.1f}s"})
        cards.append({"label": "Skill Level", "value": predictions.get("skill_level", "?").title()})

    cards_html = _render_cards(cards)

    # Flagged metrics table
    flagged_cols = ["phase", "metric", "measured", "optimal_min", "optimal_max", "deviation", "flag"]
    for r in flagged:
        r.setdefault("phase", "")
    flagged_html = _render_table(flagged, flagged_cols)

    # Recommendations
    recs: List[str] = []
    if flagged:
        sig = [r for r in flagged if r.get("flag") in ("SIGNIFICANT", "CRITICAL")]
        if sig:
            recs.append(f"{len(sig)} significant/critical deviation(s) need attention.")
    if predictions and predictions.get("recommendations"):
        recs.extend(predictions["recommendations"][:5])
    recs_html = _render_list(recs)

    # Prediction cards
    pred_cards_html = ""
    if predictions:
        imp = predictions.get("improvement_potential", {})
        pred_cards = [
            {"label": "Skill Level", "value": predictions.get("skill_level", "?").title(),
             "sub": f"confidence: {predictions.get('skill_confidence', 0)*100:.0f}%"},
            {"label": "Est. 50m", "value": f"{predictions.get('estimated_50m_time_sec', 0):.1f}s"},
            {"label": "Est. 100m", "value": f"{predictions.get('estimated_100m_time_sec', 0):.1f}s"},
            {"label": "Improvement Potential", "value": f"{imp.get('total_potential_improvement_sec', 0):.2f}s",
             "sub": f"{imp.get('num_fixable_deviations', 0)} fixable deviations"},
        ]
        pred_cards_html = _render_cards(pred_cards)

    # Entry analysis cards
    entry_html = ""
    if entry_analysis:
        ecards = []
        splash = entry_analysis.get("splash_score", {})
        if splash.get("splash_score"):
            ecards.append({"label": "Splash Score", "value": f"{splash['splash_score']:.0f}", "sub": splash.get("label", "")})
        sl = entry_analysis.get("streamline_quality", {})
        if sl.get("streamline_quality_score"):
            ecards.append({"label": "Streamline", "value": f"{sl['streamline_quality_score']:.0f}", "sub": sl.get("streamline_label", "")})
        vr = entry_analysis.get("velocity_retention", {})
        if vr.get("velocity_retention_pct"):
            ecards.append({"label": "Velocity Retention", "value": f"{vr['velocity_retention_pct']:.0f}%"})
        if ecards:
            entry_html = f'<div class="section"><h2>Entry Analysis</h2><div class="cards">{_render_cards(ecards)}</div></div>'

    # Glide analysis cards
    glide_html = ""
    if glide_analysis:
        gcards = []
        depth = glide_analysis.get("depth", {})
        if depth.get("depth_stability"):
            gcards.append({"label": "Depth Stability", "value": depth["depth_stability"]})
        lat = glide_analysis.get("lateral_deviation", {})
        if lat.get("drift_classification"):
            gcards.append({"label": "Lateral Drift", "value": lat["drift_classification"]})
        se = glide_analysis.get("streamline_effectiveness", {})
        if se.get("glide_streamline_score"):
            gcards.append({"label": "Glide Streamline", "value": f"{se['glide_streamline_score']:.0f}",
                           "sub": se.get("glide_streamline_label", "")})
        if gcards:
            glide_html = f'<div class="section"><h2>Underwater Glide</h2><div class="cards">{_render_cards(gcards)}</div></div>'

    risk_span = f'&nbsp;|&nbsp; Injury Risk: <span class="risk-{risk_level}">{risk_level}</span>' if risk_level else ""
    rt_span = f'&nbsp;|&nbsp; Reaction: {reaction_time_ms:.0f}ms' if reaction_time_ms is not None else ""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>SwimVision Coach Report — {clip_id}</title>{CSS}</head><body>
<div class="header"><h1>SwimVision Coach Report</h1>
<div class="meta"><strong>Clip:</strong> {clip_id} &nbsp;|&nbsp; <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')} &nbsp;|&nbsp; <strong>Swimmer:</strong> {swimmer_id}</div>
<div class="meta" style="margin-top:.5rem">Overall Severity: <span class="severity severity-{overall_severity}">{overall_severity}</span>{risk_span}{rt_span}</div></div>
<div class="section"><h2>Key Metrics</h2><div class="cards">{cards_html}</div></div>
{"<div class=section><h2>Flagged Deviations</h2><table><thead><tr><th>Phase</th><th>Metric</th><th>Measured</th><th>Opt Min</th><th>Opt Max</th><th>Deviation</th><th>Flag</th></tr></thead><tbody>" + flagged_html + "</tbody></table></div>" if flagged else ""}
{"<div class=section><h2>Recommendations</h2>" + recs_html + "</div>" if recs_html else ""}
{"<div class=section><h2>Performance Prediction</h2><div class=cards>" + pred_cards_html + "</div></div>" if pred_cards_html else ""}
{entry_html}
{glide_html}
<div class="footer">Generated by SwimVision on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""


# ── Athlete Feedback ──────────────────────────────────────────

def generate_athlete_report(
    clip_id: str,
    overall_score: float,
    strengths: List[str],
    focus_areas: List[str],
    skill_level: str = "intermediate",
    skill_confidence: float = 0.5,
    est_50m: Optional[float] = None,
    improvement_potential: Optional[float] = None,
) -> str:
    """Generate an athlete-friendly HTML feedback report."""
    score = int(overall_score)
    if score >= 80:
        bar_color = "#4caf50"
    elif score >= 60:
        bar_color = "#ffc107"
    else:
        bar_color = "#ff9800"

    strengths_html = _render_tips(strengths, "good")
    focus_html = _render_tips(focus_areas, "focus")

    skill_html = ""
    if skill_level:
        parts = [f'<div class="section"><h2>Your Level</h2><p style="font-size:1.2rem"><strong>{skill_level.title()}</strong> <span style="color:#6688aa">({skill_confidence*100:.0f}% confidence)</span></p>']
        if est_50m:
            parts.append(f'<p style="margin-top:.5rem;color:#8899aa">Est. 50m: {est_50m:.1f}s')
            if improvement_potential:
                parts.append(f' &nbsp;|&nbsp; Potential improvement: {improvement_potential:.2f}s')
            parts.append("</p>")
        parts.append("</div>")
        skill_html = "".join(parts)

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>SwimVision Athlete Feedback — {clip_id}</title>{CSS}</head><body>
<div class="header"><h1 style="text-align:center">Your SwimVision Feedback</h1>
<div class="score">{score}/100</div>
<div class="bar"><div class="bar-fill" style="width:{score}%;background:{bar_color}"></div></div>
<p style="color:#6688aa;text-align:center;margin-top:.5rem">{datetime.now().strftime('%Y-%m-%d')}</p></div>
{"<div class=section><h2>What You're Doing Well</h2>" + strengths_html + "</div>" if strengths_html else ""}
{"<div class=section><h2>Areas to Focus On</h2>" + focus_html + "</div>" if focus_html else ""}
{skill_html}
<div class="footer">Generated by SwimVision &middot; Keep pushing!</div>
</body></html>"""


# ── Research Export ───────────────────────────────────────────

def generate_research_export(
    clip_id: str,
    deviations: Dict[str, Any],
    analysis_mode: str = "dive",
    raw_json: Optional[str] = None,
) -> str:
    """Generate a research-oriented HTML data export."""
    all_rows: List[Dict[str, Any]] = []
    for phase_name in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle"):
        for row in deviations.get(phase_name, []):
            if isinstance(row, dict):
                r = dict(row)
                r["phase"] = phase_name.replace("_phase", "").replace("_cycle", "")
                all_rows.append(r)

    cols = ["phase", "metric", "measured", "optimal_min", "optimal_max", "deviation", "flag"]
    table_html = _render_table(all_rows, cols)
    raw_html = f"<h2>Raw Analysis Data</h2><pre>{raw_json}</pre>" if raw_json else ""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>SwimVision Research Export — {clip_id}</title>{CSS}</head><body>
<h1 style="color:#58a6ff;font-size:1.4rem;margin-bottom:1.5rem">SwimVision Research Export</h1>
<div class="meta">Clip: {clip_id} &nbsp;|&nbsp; Date: {datetime.now().strftime('%Y-%m-%d')} &nbsp;|&nbsp; Mode: {analysis_mode}</div>
<h2 style="color:#79c0ff;margin:1.5rem 0 .75rem">Deviations</h2>
<table><thead><tr><th>Phase</th><th>Metric</th><th>Measured</th><th>Opt Min</th><th>Opt Max</th><th>Deviation</th><th>Flag</th></tr></thead><tbody>{table_html}</tbody></table>
{raw_html}
<div class="footer">SwimVision Research Export &middot; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""


# ── CLI ───────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for HTML report generation."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate SwimVision HTML reports.")
    parser.add_argument("--deviations", required=True, help="Deviations JSON path.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--template", choices=["coach", "athlete", "research"], default="coach")
    parser.add_argument("--clip-id", default="unknown", help="Clip identifier.")
    parser.add_argument("--swimmer-id", default="unknown", help="Swimmer identifier.")
    parser.add_argument("--predictions", help="Performance prediction JSON path.")
    parser.add_argument("--entry-analysis", help="Entry analysis JSON path.")
    parser.add_argument("--glide-analysis", help="Glide analysis JSON path.")
    parser.add_argument("--reaction-time-ms", type=float, help="Reaction time in ms.")
    parser.add_argument("--risk-level", help="Injury risk level.")
    parser.add_argument("--overall-score", type=float, default=75, help="Overall athlete score 0-100.")
    parser.add_argument("--strengths", nargs="*", default=[], help="Athlete strengths.")
    parser.add_argument("--focus-areas", nargs="*", default=[], help="Athlete focus areas.")
    parser.add_argument("--skill-level", default="intermediate", help="Skill classification.")
    parser.add_argument("--skill-confidence", type=float, default=0.5, help="Skill confidence 0-1.")
    parser.add_argument("--est-50m", type=float, help="Estimated 50m time.")
    parser.add_argument("--improvement-potential", type=float, help="Improvement potential in seconds.")
    return parser


def main() -> int:
    """Run HTML report CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        with open(args.deviations, "r", encoding="utf-8") as f:
            deviations = json.load(f)
    except Exception as exc:
        LOGGER.error("Failed to load deviations: %s", exc)
        return 1

    # Load optional data
    predictions = None
    if args.predictions:
        try:
            with open(args.predictions, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        except Exception:
            pass

    entry_analysis = None
    if args.entry_analysis:
        try:
            with open(args.entry_analysis, "r", encoding="utf-8") as f:
                entry_analysis = json.load(f)
        except Exception:
            pass

    glide_analysis = None
    if args.glide_analysis:
        try:
            with open(args.glide_analysis, "r", encoding="utf-8") as f:
                glide_analysis = json.load(f)
        except Exception:
            pass

    if args.template == "coach":
        html = generate_coach_report(
            args.clip_id, deviations, swimmer_id=args.swimmer_id,
            predictions=predictions, entry_analysis=entry_analysis,
            glide_analysis=glide_analysis,
            reaction_time_ms=args.reaction_time_ms,
            risk_level=args.risk_level,
        )
    elif args.template == "athlete":
        html = generate_athlete_report(
            args.clip_id, overall_score=args.overall_score,
            strengths=args.strengths, focus_areas=args.focus_areas,
            skill_level=args.skill_level, skill_confidence=args.skill_confidence,
            est_50m=args.est_50m, improvement_potential=args.improvement_potential,
        )
    else:
        html = generate_research_export(args.clip_id, deviations)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    LOGGER.info("Saved %s HTML report to %s", args.template, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
