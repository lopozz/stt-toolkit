import difflib
import html
import json
from pathlib import Path
from typing import Any

import streamlit as st


def _load_one_result(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metadata = payload.get("metadata", {})
    results = payload.get("results", {})
    samples = results.get("samples", [])

    if not isinstance(metadata, dict) or not isinstance(results, dict):
        return None
    if not isinstance(samples, list):
        return None

    model = str(metadata.get("model", "unknown"))
    dataset = str(metadata.get("dataset", "unknown"))
    avg_wer = results.get("wer")

    return {
        "path": str(path),
        "model": model,
        "dataset": dataset,
        "avg_wer": avg_wer,
        "samples": samples,
    }


def _load_results(root_dir: str) -> list[dict[str, Any]]:
    root = Path(root_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    loaded: list[dict[str, Any]] = []
    for file_path in sorted(root.rglob("*.json")):
        item = _load_one_result(file_path)
        if item is not None:
            loaded.append(item)
    return loaded


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_typo_like(ref_token: str, pred_token: str, threshold: float = 0.72) -> bool:
    ref_token = ref_token.strip().lower()
    pred_token = pred_token.strip().lower()

    if not ref_token or not pred_token:
        return False

    ratio = difflib.SequenceMatcher(a=ref_token, b=pred_token).ratio()
    return ratio >= threshold


def _inject_ui_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1 {
            font-size: 2.5rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.35rem !important;
        }

        h2, h3 {
            font-size: 1.85rem !important;
            line-height: 1.2 !important;
            margin-top: 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }

        p, label, .stCaption, .stMarkdown, .stText {
            font-size: 1.02rem !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            font-size: 1.02rem !important;
            line-height: 1.55 !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            min-height: 3rem !important;
            font-size: 1rem !important;
        }

        div[data-testid="stSlider"] {
            padding-top: 0.35rem;
        }

        .wer-muted {
            color: #B7BDC9;
        }

        .wer-card {
            background: #171A23;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .wer-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 0.75rem 0 1rem 0;
        }

        .wer-stat-card {
            background: #171A23;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .wer-stat-label {
            font-size: 0.92rem;
            color: #A8B0BF;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }

        .wer-stat-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.35;
        }

        .wer-subtle-badge {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.06);
            font-family: monospace;
            font-size: 0.98rem;
        }

        @media (max-width: 900px) {
            .wer-stat-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 2rem !important;
            }

            h2, h3 {
                font-size: 1.5rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _token_badge(
    token: str,
    kind: str,
    title: str,
    ref_token: str | None = None,
) -> str:
    token_html = html.escape(token)
    title_html = html.escape(title)

    if kind == "equal":
        return (
            f"<span title='{title_html}' style='"
            "color:#F3F4F6;"
            "font-size:1.08rem;"
            "line-height:1.7;'>"
            f"{token_html}</span>"
        )

    base_style = (
        "display:inline-block;"
        "margin:0 0.16rem 0.1rem 0;"
        "padding:0.08rem 0.38rem;"
        "border-radius:0.5rem;"
        "font-weight:600;"
        "font-size:0.98rem;"
        "line-height:1.45;"
        "vertical-align:baseline;"
    )

    if kind == "added":
        return (
            f"<span title='{title_html}' style='" + base_style + "background:#E8F1FB;"
            "color:#0F172A;"
            "border:2px solid #1D4ED8;'>"
            f"+ {token_html}</span>"
        )

    if kind == "missing":
        return (
            f"<span title='{title_html}' style='" + base_style + "background:#F3F4F6;"
            "color:#111827;"
            "border:2px dashed #4B5563;"
            "font-style:italic;'>"
            f"∅ {token_html}</span>"
        )

    if kind == "typo":
        ref_suffix = ""
        if ref_token:
            ref_suffix = (
                f"<span style='font-weight:500; opacity:0.82;'>"
                f" ← {html.escape(ref_token)}</span>"
            )
        return (
            f"<span title='{title_html}' style='" + base_style + "background:#FFF4DB;"
            "color:#111827;"
            "border:2px dotted #B45309;'>"
            f"≈ {token_html}{ref_suffix}</span>"
        )

    if kind == "wrong":
        ref_suffix = ""
        if ref_token:
            ref_suffix = (
                f"<span style='font-weight:500; opacity:0.82;'>"
                f" ← {html.escape(ref_token)}</span>"
            )
        return (
            f"<span title='{title_html}' style='" + base_style + "background:#F7E8F6;"
            "color:#111827;"
            "border:2px solid #A21CAF;'>"
            f"≠ {token_html}{ref_suffix}</span>"
        )

    return f"<span>{token_html}</span>"


def _panel_shell_html(title: str, body_html: str, min_height: int = 240) -> str:
    return (
        "<div style='display:flex; flex-direction:column; gap:10px;'>"
        f"<div style='font-size:1.9rem; font-weight:700; color:#F8FAFC; line-height:1.15;'>{html.escape(title)}</div>"
        "<div style='"
        f"min-height:{min_height}px;"
        "padding:18px 18px;"
        "border-radius:14px;"
        "background:#262730;"
        "color:#FAFAFA;"
        "border:1px solid rgba(255,255,255,0.08);"
        "box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);'>"
        f"{body_html}"
        "</div>"
        "</div>"
    )


def _reference_panel_html(reference: str) -> str:
    escaped = html.escape(reference)
    body = (
        "<div style='"
        "font-size:1.15rem;"
        "line-height:1.9;"
        "color:#F8FAFC;"
        "white-space:pre-wrap;"
        "word-break:break-word;'>"
        f"{escaped}"
        "</div>"
    )
    return _panel_shell_html("Reference", body, min_height=240)


def _prediction_diff_html(reference: str, prediction: str) -> str:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=pred_tokens)

    chunks: list[str] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk_tokens = ref_tokens[i1:i2]
        pred_chunk_tokens = pred_tokens[j1:j2]

        if opcode == "equal":
            for tok in pred_chunk_tokens:
                chunks.append(_token_badge(tok, "equal", "Correct token"))

        elif opcode == "insert":
            for tok in pred_chunk_tokens:
                chunks.append(_token_badge(tok, "added", "Added word in prediction"))

        elif opcode == "delete":
            for tok in ref_chunk_tokens:
                chunks.append(
                    _token_badge(tok, "missing", "Missing word from reference")
                )

        elif opcode == "replace":
            common_len = min(len(ref_chunk_tokens), len(pred_chunk_tokens))

            for k in range(common_len):
                ref_tok = ref_chunk_tokens[k]
                pred_tok = pred_chunk_tokens[k]

                if _is_typo_like(ref_tok, pred_tok):
                    chunks.append(
                        _token_badge(
                            pred_tok,
                            "typo",
                            "Typo-like substitution",
                            ref_token=ref_tok,
                        )
                    )
                else:
                    chunks.append(
                        _token_badge(
                            pred_tok,
                            "wrong",
                            "Wrong substituted word",
                            ref_token=ref_tok,
                        )
                    )

            for tok in pred_chunk_tokens[common_len:]:
                chunks.append(_token_badge(tok, "added", "Added word in prediction"))

            for tok in ref_chunk_tokens[common_len:]:
                chunks.append(
                    _token_badge(tok, "missing", "Missing word from reference")
                )

    body = (
        "<div style='"
        "font-size:1.08rem;"
        "line-height:1.75;"
        "color:#F8FAFC;"
        "white-space:normal;"
        "word-break:break-word;'>" + " ".join(chunks) + "</div>"
    )
    return _panel_shell_html("Prediction", body, min_height=240)


def _legend_html() -> str:
    return (
        "<div style='"
        "margin:10px 0 18px 0;"
        "padding:14px 16px;"
        "border-radius:14px;"
        "background:#171A23;"
        "border:1px solid rgba(255,255,255,0.08);"
        "display:flex;"
        "flex-wrap:wrap;"
        "gap:12px;"
        "align-items:center;'>"
        "<span style='font-size:1rem; color:#CBD5E1; font-weight:700; margin-right:4px;'>Legend</span>"
        "<span style='display:inline-block; padding:6px 11px; border-radius:9px; background:#E8F1FB; color:#111827; border:2px solid #1D4ED8; font-weight:700; font-size:1rem;'>+ Added</span>"
        "<span style='display:inline-block; padding:6px 11px; border-radius:9px; background:#F3F4F6; color:#111827; border:2px dashed #4B5563; font-style:italic; font-weight:700; font-size:1rem;'>∅ Missing</span>"
        "<span style='display:inline-block; padding:6px 11px; border-radius:9px; background:#FFF4DB; color:#111827; border:2px dotted #B45309; font-weight:700; font-size:1rem;'>≈ Typo</span>"
        "<span style='display:inline-block; padding:6px 11px; border-radius:9px; background:#F7E8F6; color:#111827; border:2px solid #A21CAF; font-weight:700; font-size:1rem;'>≠ Wrong word</span>"
        "</div>"
    )


def _run_summary_html(model: str, dataset: str) -> str:
    return (
        "<div class='wer-card' style='margin-top:8px; margin-bottom:10px;'>"
        "<div style='display:flex; flex-wrap:wrap; gap:18px; align-items:center;'>"
        f"<div><span class='wer-muted'><strong>Model</strong></span><br><span class='wer-subtle-badge'>{html.escape(model)}</span></div>"
        f"<div><span class='wer-muted'><strong>Dataset</strong></span><br><span class='wer-subtle-badge'>{html.escape(dataset)}</span></div>"
        "</div>"
        "</div>"
    )


def _sample_stats_html(
    sample_idx: int,
    sample_count: int,
    sample_wer: float | None,
    source_text: str,
) -> str:
    wer_text = f"{sample_wer:.4f}" if sample_wer is not None else "n/a"
    source_html = html.escape(source_text) if source_text else "n/a"

    return (
        "<div class='wer-stat-grid'>"
        "<div class='wer-stat-card'>"
        "<div class='wer-stat-label'>Sample</div>"
        f"<div class='wer-stat-value'>{sample_idx + 1}/{sample_count}</div>"
        "</div>"
        "<div class='wer-stat-card'>"
        "<div class='wer-stat-label'>Sample WER</div>"
        f"<div class='wer-stat-value'>{wer_text}</div>"
        "</div>"
        "<div class='wer-stat-card'>"
        "<div class='wer-stat-label'>Source</div>"
        f"<div class='wer-stat-value' style='font-size:1.05rem; font-family:monospace;'>{source_html}</div>"
        "</div>"
        "</div>"
    )


def main() -> None:
    st.set_page_config(page_title="WER Results Viewer", layout="wide")
    _inject_ui_css()

    st.title("WER Results Viewer")
    st.caption("Inspect transcription mistakes across model and dataset runs.")

    default_dir = "results/wer_bench"

    runs = _load_results(default_dir)
    if not runs:
        st.warning("No valid JSON result files found in this directory.")
        return

    models = sorted({r["model"] for r in runs})
    datasets = sorted({r["dataset"] for r in runs})

    filter_left, filter_right = st.columns(2)
    with filter_left:
        selected_models = st.multiselect(
            "Filter by model",
            models,
            default=models,
        )
    with filter_right:
        selected_datasets = st.multiselect(
            "Filter by dataset",
            datasets,
            default=datasets,
        )

    filtered_runs = [
        run
        for run in runs
        if run["model"] in selected_models and run["dataset"] in selected_datasets
    ]

    if not filtered_runs:
        st.warning("No runs match the selected filters.")
        return

    run_labels = []
    for run in filtered_runs:
        overall = _safe_float(run["avg_wer"])
        overall_text = f"{overall:.4f}" if overall is not None else "n/a"
        run_labels.append(f"{run['model']} | {run['dataset']} | WER {overall_text}")

    selected_label = st.selectbox(
        "Select run",
        options=run_labels,
        index=0,
    )
    selected_run = filtered_runs[run_labels.index(selected_label)]

    st.markdown(
        _run_summary_html(
            model=selected_run["model"],
            dataset=selected_run["dataset"],
        ),
        unsafe_allow_html=True,
    )

    sample_list = selected_run["samples"]
    if not sample_list:
        st.info("This run has no samples.")
        return

    st.markdown("### Sample Navigator")
    idx = st.slider("Sample index", 0, len(sample_list) - 1, 0, 1)
    sample = sample_list[idx]

    sample_wer = _safe_float(sample.get("wer"))
    source_text = str(sample.get("source", ""))
    ref_text = str(sample.get("ref", ""))
    pred_text = str(sample.get("pred", ""))

    st.markdown(
        _sample_stats_html(
            sample_idx=idx,
            sample_count=len(sample_list),
            sample_wer=sample_wer,
            source_text=source_text,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(_legend_html(), unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(_reference_panel_html(ref_text), unsafe_allow_html=True)

    with right:
        st.markdown(
            _prediction_diff_html(ref_text, pred_text),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
