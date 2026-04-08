"""Interactive Streamlit dashboard for NLP Sentiment Intelligence.

Provides live prediction with SHAP explanations, model comparison arena,
explainability deep-dive, and production drift monitoring.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.text_processing import normalize_text

config = get_config()
METRICS_DIR = PROJECT_ROOT / config.paths.metrics_dir
FIGURES_DIR = PROJECT_ROOT / config.paths.reports_dir

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="NLP Sentiment Intelligence",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 0.95rem; }

h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.3rem !important; }
h3 { font-size: 1.15rem !important; }
h4 { font-size: 1.05rem !important; }

.metric-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 0.7rem 0.8rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card h3 {
    color: #a0a0b0;
    font-size: 0.75rem !important;
    margin: 0 0 0.2rem 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.3;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 16px;
    font-weight: 600;
    font-size: 0.8rem;
}
.badge-green  { background: rgba(44,160,44,0.15); color: #2ca02c; border: 1px solid rgba(44,160,44,0.3); }
.badge-yellow { background: rgba(180,140,40,0.15); color: #b08c28; border: 1px solid rgba(180,140,40,0.3); }
.badge-red    { background: rgba(180,60,60,0.15); color: #b43c3c; border: 1px solid rgba(180,60,60,0.3); }

.insight-box {
    background: #1e1e2e;
    border-left: 3px solid #4a6fa5;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin: 0.5rem 0;
    color: #cdd6f4;
    font-size: 0.88rem;
    line-height: 1.5;
}

.word-highlight { padding: 1px 3px; border-radius: 3px; margin: 0 1px; }

div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid #4a6fa5 !important;
    font-weight: 600;
}

div[data-testid="stDataFrame"] { font-size: 0.85rem !important; }
div[data-testid="stDataFrame"] table { font-size: 0.85rem !important; }

section[data-testid="stSidebar"] { font-size: 0.85rem; }
section[data-testid="stSidebar"] .stMarkdown p { font-size: 0.85rem; }

button[kind="primary"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}
button[kind="primary"]:hover {
    background-color: #185d8c !important;
    border-color: #185d8c !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _latest_json(pattern: str) -> Path | None:
    """Find the most recently modified JSON matching a glob pattern."""
    files = sorted(METRICS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_json(pattern: str) -> dict | None:
    """Load the latest JSON file matching a glob pattern."""
    path = _latest_json(pattern)
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def _metric_card(title: str, value: str, color: str = "#cdd6f4") -> str:
    """Generate HTML for a styled metric card."""
    return (
        f'<div class="metric-card">'
        f'<h3>{title}</h3>'
        f'<p class="value" style="color:{color}">{value}</p>'
        f"</div>"
    )


def _badge(text: str, level: str) -> str:
    """Generate HTML for a colored badge."""
    css_class = {
        "high": "badge-green",
        "medium": "badge-yellow",
        "low": "badge-red",
    }.get(level, "badge-green")
    return f'<span class="badge {css_class}">{text}</span>'


def _check_language_coverage(text: str) -> float:
    """Return fraction of words in the TF-IDF vocabulary."""
    normalized = normalize_text(text)
    words = normalized.split()
    if not words:
        return 0.0
    vocab = set(vectorizer.get_feature_names_out())
    matched = sum(1 for w in words if w in vocab)
    return matched / len(words)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load champion model and extract components for SHAP."""
    model_path = PROJECT_ROOT / config.api.model_path
    mdl = joblib.load(model_path)
    base = mdl.calibrated_classifiers_[0].estimator
    vec = base.named_steps["tfidf"]
    clf = base.named_steps["clf"]
    exp = shap.LinearExplainer(clf, vec.transform([""]))
    return mdl, vec, clf, exp


@st.cache_data
def load_all_metrics():
    """Load all metric JSONs into a dictionary."""
    return {
        "test_results": _load_json("phase2_test_results_*.json"),
        "business": _load_json("phase2_business_tradeoff_*.json"),
        "shap_results": _load_json("phase2_shap_results_*.json"),
        "shap_highlights": _load_json("phase2_shap_word_highlights_*.json"),
        "drift": _load_json("drift_summary_*.json"),
    }


model, vectorizer, classifier, explainer = load_model()
metrics = load_all_metrics()


def _get_routing(prob: float) -> tuple[str, str, str]:
    """Return confidence level, routing action, and explanation."""
    if prob > 0.85 or prob < 0.15:
        return (
            "high",
            "auto_classify",
            "Model is confident. No human review needed.",
        )
    if prob > 0.60 or prob < 0.40:
        return (
            "medium",
            "human_review",
            "Mixed signals detected. Routed to human reviewer.",
        )
    return (
        "low",
        "escalate",
        "Model uncertain. Escalated to senior analyst.",
    )


def _compute_shap_for_text(
    text: str,
) -> tuple[dict[str, float], list[tuple[str, float]], list[tuple[str, float]]]:
    """Compute per-word SHAP values and top positive/negative contributors."""
    normalized = normalize_text(text)
    x_tfidf = vectorizer.transform([normalized])
    shap_values = explainer.shap_values(x_tfidf)

    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_idx = x_tfidf.nonzero()[1]
    values = shap_values[0, nonzero_idx]
    names = feature_names[nonzero_idx]

    word_shap: dict[str, float] = {}
    for name, val in zip(names, values):
        for token in name.split():
            if token in word_shap:
                word_shap[token] = max(
                    word_shap[token], float(val), key=abs,
                )
            else:
                word_shap[token] = float(val)

    feature_map = dict(zip(names, values))
    sorted_feats = sorted(feature_map.items(), key=lambda x: x[1])
    top_neg = [(n, round(float(v), 4)) for n, v in sorted_feats if v < 0][:5]
    top_pos = [
        (n, round(float(v), 4))
        for n, v in reversed(sorted_feats)
        if v > 0
    ][:5]

    return word_shap, top_pos, top_neg


def _render_highlighted_text(text: str, word_shap: dict[str, float]) -> str:
    """Render review text with SHAP-colored word highlights."""
    normalized = normalize_text(text)
    words = normalized.split()
    abs_vals = [abs(v) for v in word_shap.values()] if word_shap else [1.0]
    max_shap = max(abs_vals) if abs_vals else 1.0

    html_parts = []
    for word in words:
        if word in word_shap:
            val = word_shap[word]
            intensity = min(abs(val) / max_shap, 1.0) * 0.6
            if val > 0:
                bg = f"rgba(0,255,0,{intensity})"
            else:
                bg = f"rgba(255,0,0,{intensity})"
            html_parts.append(
                f'<span class="word-highlight" style="background-color:{bg}">'
                f"{word}</span>"
            )
        else:
            html_parts.append(word)
    return " ".join(html_parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### NLP Sentiment Intelligence")

    test_results = metrics.get("test_results") or {}
    champion_key = test_results.get("champion", "lr_tuned_calibrated")
    champion_metrics = test_results.get(champion_key, {})
    champion_f1 = champion_metrics.get("f1", 0)

    st.caption(f"Version: {config.project.version}")
    st.caption(f"Champion: {Path(config.api.model_path).stem}")
    st.caption(f"Test F1: {champion_f1:.4f}")

    st.markdown("---")
    st.markdown(
        " | "
        "[GitHub](https://github.com/marianunez-data/nlp-sentiment-intelligence)"
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction",
    "Model Comparison",
    "Explainability",
    "Production Monitor",
])

FOOTER_HTML = (
    '<div style="text-align: center; padding: 20px; margin-top: 40px; '
    'border-top: 1px solid #333; color: #888; font-size: 0.8rem;">'
    "<strong>Maria Camila Gonzalez Nu\u00f1ez</strong><br>"
    '<a href="https://www.linkedin.com/in/marianunez-data" '
    'target="_blank" style="color: #1f77b4;">LinkedIn</a> | '
    '<a href="https://github.com/marianunez-data/nlp-sentiment-intelligence" '
    'target="_blank" style="color: #1f77b4;">GitHub</a><br>'
    '<span style="color: #666;">Built with Python &middot; '
    "scikit-learn &middot; FastAPI &middot; Streamlit &middot; "
    "SHAP &middot; Evidently AI &middot; Docker</span>"
    "</div>"
)

# ===== TAB 1 -- LIVE PREDICTION =====
_EXAMPLE_POSITIVE = (
    "Amazing film, one of the best movies I have ever seen. "
    "The acting was superb and the story was incredible."
)
_EXAMPLE_NEGATIVE = (
    "This was a terrible movie. The acting was awful, the plot "
    "made no sense, and it was a complete waste of time."
)
_EXAMPLE_BORDERLINE = (
    "The film has excellent cinematography and great performances "
    "but the plot was bad and the ending was terrible."
)

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

with tab1:
    st.markdown("### Live Sentiment Analysis")

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("Try: Positive", use_container_width=True):
            st.session_state.review_text = _EXAMPLE_POSITIVE
    with ex2:
        if st.button("Try: Negative", use_container_width=True):
            st.session_state.review_text = _EXAMPLE_NEGATIVE
    with ex3:
        if st.button("Try: Borderline", use_container_width=True):
            st.session_state.review_text = _EXAMPLE_BORDERLINE

    review_input = st.text_area(
        "Enter a movie review",
        value=st.session_state.review_text,
        height=120,
        placeholder=(
            "e.g. This movie was a masterpiece. The acting was superb and "
            "the storyline kept me on the edge of my seat the entire time."
        ),
    )
    st.caption("This model was trained on English movie reviews. Non-English input may produce unreliable results.")


    if st.button("Analyze", type="primary", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter a review to analyze.")
        else:
            # Language coverage check
            coverage = _check_language_coverage(review_input)
            if coverage < 0.20:
                st.warning(
                    "This model was trained on English reviews only. "
                    "Results may not be accurate for other languages."
                )

            normalized = normalize_text(review_input)
            prob = float(model.predict_proba([normalized])[0, 1])
            sentiment = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
            sent_color = "#2ca02c" if prob >= 0.5 else "#b43c3c"
            conf_level, routing, explanation = _get_routing(prob)
            word_shap, top_pos, top_neg = _compute_shap_for_text(
                review_input,
            )

            # Row 1: sentiment + probability + gauge
            r1c1, r1c2 = st.columns([1, 2])
            with r1c1:
                st.markdown(
                    _metric_card("Sentiment", sentiment, sent_color),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _metric_card(
                        "Probability", f"{prob:.4f}", "#cdd6f4",
                    ),
                    unsafe_allow_html=True,
                )
            with r1c2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": sent_color},
                        "steps": [
                            {
                                "range": [0, 0.4],
                                "color": "rgba(180,60,60,0.1)",
                            },
                            {
                                "range": [0.4, 0.6],
                                "color": "rgba(180,140,40,0.1)",
                            },
                            {
                                "range": [0.6, 1],
                                "color": "rgba(44,160,44,0.1)",
                            },
                        ],
                    },
                    number={"valueformat": ".3f"},
                ))
                gauge.update_layout(
                    height=180,
                    margin={"t": 20, "b": 5, "l": 20, "r": 20},
                )
                st.plotly_chart(gauge, use_container_width=True)

            # Row 2: routing info
            routing_color = {
                "auto_classify": "#2ca02c",
                "human_review": "#b08c28",
                "escalate": "#b43c3c",
            }.get(routing, "#cdd6f4")

            rc1, rc2 = st.columns(2)
            with rc1:
                badge_html = _badge(conf_level.upper(), conf_level)
                st.markdown(
                    _metric_card("Confidence", badge_html, "#cdd6f4"),
                    unsafe_allow_html=True,
                )
            with rc2:
                st.markdown(
                    _metric_card(
                        "Routing",
                        (
                            f'<span style="color:{routing_color}">'
                            f"{routing}</span>"
                        ),
                        routing_color,
                    ),
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="insight-box">{explanation}</div>',
                unsafe_allow_html=True,
            )

            # Row 3: SHAP chart
            all_words = top_pos + top_neg
            if all_words:
                all_words_sorted = sorted(all_words, key=lambda x: x[1])
                fig_shap = go.Figure(go.Bar(
                    y=[w[0] for w in all_words_sorted],
                    x=[w[1] for w in all_words_sorted],
                    orientation="h",
                    marker_color=[
                        "#2ca02c" if v > 0 else "#b43c3c"
                        for _, v in all_words_sorted
                    ],
                ))
                fig_shap.update_layout(
                    title="Top Contributing Words (SHAP)",
                    template="plotly_white",
                    height=280,
                    margin={"t": 30, "b": 10, "l": 10, "r": 10},
                    xaxis_title="SHAP value",
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            # Row 4: word highlighting
            st.markdown("#### SHAP Word Highlighting")
            highlighted = _render_highlighted_text(review_input, word_shap)
            st.markdown(
                f'<div style="line-height:1.8; font-size:0.9rem;">'
                f"{highlighted}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Green = pushes toward positive | "
                "Red = pushes toward negative | "
                "Intensity = SHAP magnitude"
            )

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ===== TAB 2 -- MODEL COMPARISON =====
with tab2:
    st.markdown("### 10 Models. 1 Champion. Here's the evidence.")
    test_results = metrics.get("test_results")

    if test_results is None:
        st.warning("Test results JSON not found.")
    else:
        skip_keys = {
            "champion", "inference_times",
            "mcnemar_champion_vs_baseline",
            "mcnemar_champion_vs_bert",
            "mcnemar_lr_default_vs_tuned",
        }
        champion_name = test_results.get("champion", "lr_tuned_calibrated")

        model_rows = []
        for key, vals in test_results.items():
            if key in skip_keys or not isinstance(vals, dict):
                continue
            if "f1" not in vals:
                continue
            model_rows.append({
                "Model": key,
                "F1": vals.get("f1", 0),
                "F1 CI": (
                    f"[{vals['f1_ci'][0]:.4f}, {vals['f1_ci'][1]:.4f}]"
                    if vals.get("f1_ci") else "-"
                ),
                "ROC-AUC": vals.get("roc_auc", 0),
                "Precision": vals.get("precision", 0),
                "Recall": vals.get("recall", 0),
                "ms/pred": vals.get("ms_per_prediction", 0),
            })

        if model_rows:
            import pandas as pd

            df_models = pd.DataFrame(model_rows).sort_values(
                "F1", ascending=False,
            )

            def _highlight_champion(row):
                if row["Model"] == champion_name:
                    return [
                        "background: rgba(44,160,44,0.12)",
                    ] * len(row)
                return [""] * len(row)

            styled = df_models.style.apply(
                _highlight_champion, axis=1,
            ).format({
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "ms/pred": "{:.2f}",
            })
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=380,
            )

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**F1 Score Comparison**")
                df_sorted = df_models.sort_values("F1")
                colors = [
                    "#2ca02c" if m == champion_name else "#4a6fa5"
                    for m in df_sorted["Model"]
                ]
                fig_f1 = go.Figure(go.Bar(
                    y=df_sorted["Model"],
                    x=df_sorted["F1"],
                    orientation="h",
                    marker_color=colors,
                    text=df_sorted["F1"].apply(lambda v: f"{v:.4f}"),
                    textposition="outside",
                ))
                fig_f1.update_layout(
                    template="plotly_white",
                    height=350,
                    margin={"l": 10, "r": 50, "t": 10, "b": 10},
                    xaxis={"range": [0, 1]},
                )
                st.plotly_chart(fig_f1, use_container_width=True)

            with col_b:
                st.markdown("**Inference Latency (ms/prediction)**")
                df_lat = df_models[
                    df_models["ms/pred"] > 0
                ].sort_values("ms/pred")
                lat_colors = [
                    "#b43c3c" if "distilbert" in m.lower() else "#4a6fa5"
                    for m in df_lat["Model"]
                ]
                fig_lat = go.Figure(go.Bar(
                    y=df_lat["Model"],
                    x=df_lat["ms/pred"],
                    orientation="h",
                    marker_color=lat_colors,
                    text=df_lat["ms/pred"].apply(
                        lambda v: f"{v:.2f}",
                    ),
                    textposition="outside",
                ))
                fig_lat.update_layout(
                    template="plotly_white",
                    height=350,
                    margin={"l": 10, "r": 50, "t": 10, "b": 10},
                    xaxis_type="log",
                    xaxis_title="ms (log scale)",
                )
                st.plotly_chart(fig_lat, use_container_width=True)

            with col_c:
                st.markdown("**Cost Analysis**")
                biz = metrics.get("business") or {}
                champ_cost = biz.get("champion_monthly_cost", 50)
                bert_cost = biz.get("bert_monthly_cost", 500)
                champ_f1_val = biz.get("champion_f1", champion_f1)
                bert_f1_val = biz.get("bert_f1", 0)
                champ_cost_str = str(champ_cost).lstrip("$")
                bert_cost_str = str(bert_cost).lstrip("$")
                st.markdown(
                    _metric_card(
                        "LR Champion",
                        f"${champ_cost_str}/mo "
                        f"&mdash; F1={champ_f1_val:.4f}",
                        "#2ca02c",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _metric_card(
                        "DistilBERT",
                        f"${bert_cost_str}/mo "
                        f"&mdash; F1={bert_f1_val:.4f}",
                        "#b43c3c",
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "*Estimates for 1M reviews/month. "
                    "AWS on-demand pricing, US East, April 2026.*"
                )
                speed = biz.get("speed_ratio", 0)
                if speed:
                    st.markdown(
                        f'<div class="insight-box">'
                        f"LR is <b>{speed:.0f}x faster</b> than BERT "
                        f"at inference, with <b>higher F1</b>.</div>",
                        unsafe_allow_html=True,
                    )

        mcnemar_keys = [
            k for k in test_results if k.startswith("mcnemar_")
        ]
        if mcnemar_keys:
            st.markdown("---")
            st.markdown("#### McNemar's Statistical Significance")
            for key in mcnemar_keys:
                mc = test_results[key]
                if mc.get("significant"):
                    sig = "Significant"
                else:
                    sig = "Not significant"
                label = (
                    key.replace("mcnemar_", "")
                    .replace("_", " ")
                    .title()
                )
                st.markdown(
                    f"- **{label}**: "
                    f"chi2={mc.get('chi2', 0):.2f}, "
                    f"p={mc.get('p_value', 1):.4f} "
                    f"&rarr; {sig}"
                )

        st.markdown(
            '<div class="insight-box">'
            "<b>Key insight</b>: LR default (C=1.0) achieved identical "
            "performance to the Optuna-tuned version -- confirming the "
            "problem is linearly separable in TF-IDF space.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ===== TAB 3 -- EXPLAINABILITY =====
with tab3:
    st.markdown("### Every prediction is auditable. No black boxes.")

    shap_data = metrics.get("shap_results")
    shap_hl = metrics.get("shap_highlights")

    _NEGATIVE_WORDS = {
        "bad", "worst", "nothing", "boring", "waste", "awful",
        "the worst", "don", "terrible", "horrible", "poor",
    }
    _POSITIVE_WORDS = {
        "great", "best", "the best", "well", "excellent", "love",
        "amazing", "wonderful", "perfect", "brilliant", "superb",
    }

    def _shap_feature_color(name: str) -> str:
        """Return color based on sentiment direction of a feature."""
        if name in _NEGATIVE_WORDS:
            return "#d62728"
        if name in _POSITIVE_WORDS:
            return "#1f77b4"
        return "#888888"

    if shap_data and shap_data.get("top_20_features"):
        features = shap_data["top_20_features"]
        sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]))
        fig_top20 = go.Figure(go.Bar(
            y=[f[0] for f in sorted_feats],
            x=[f[1] for f in sorted_feats],
            orientation="h",
            marker_color=[
                _shap_feature_color(name) for name, _ in sorted_feats
            ],
        ))
        fig_top20.update_layout(
            title="Top 20 SHAP Features (Global)",
            template="plotly_white",
            height=450,
            margin={"l": 10, "r": 10, "t": 35, "b": 10},
            xaxis_title="Mean |SHAP value|",
        )
        st.plotly_chart(fig_top20, use_container_width=True)
        st.markdown(
            '<span style="color: #d62728;">&#9632;</span> Negative sentiment driver &nbsp;&nbsp;'
            '<span style="color: #1f77b4;">&#9632;</span> Positive sentiment driver &nbsp;&nbsp;'
            '<span style="color: #888;">&#9632;</span> Context-dependent (direction varies by review)',
            unsafe_allow_html=True,
        )
    else:
        st.info("SHAP results JSON not found.")

    if shap_hl:
        st.markdown("---")
        st.markdown("#### Example Predictions with Word Highlighting")
        example_keys = [
            "strong_positive", "strong_negative", "borderline",
        ]
        example_titles = [
            "Strong Positive", "Strong Negative", "Borderline",
        ]
        cols = st.columns(3)
        for col, key, title in zip(cols, example_keys, example_titles):
            with col:
                ex = shap_hl.get(key, {})
                if not ex:
                    st.info(f"No {key} example found.")
                    continue
                pred = ex.get("prediction", 0)
                actual = ex.get("actual", 0)
                pred_label = (
                    "Positive" if pred >= 0.5 else "Negative"
                )
                actual_label = (
                    "Positive" if actual == 1 else "Negative"
                )

                st.markdown(f"**{title}**")
                st.markdown(
                    f"Prediction: **{pred_label}** ({pred:.2f}) | "
                    f"Actual: **{actual_label}**"
                )

                word_contribs = ex.get("word_contributions", {})
                review_text = ex.get("review_text", "")
                if review_text and word_contribs:
                    highlighted = _render_highlighted_text(
                        review_text[:500], word_contribs,
                    )
                    st.markdown(
                        '<div style="line-height:1.7; font-size:0.82rem; '
                        'max-height:220px; overflow-y:auto;">'
                        f"{highlighted}</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        st.markdown(
            '<div class="insight-box">'
            '<b>"bad" is the #1 feature</b> -- simple but effective. '
            "A single word carries more predictive power than complex "
            "syntactic patterns.</div>",
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            '<div class="insight-box">'
            '<b>Bigrams validate design</b> -- "the worst" and "the '
            'best" appear in the top 20, confirming that '
            "<code>ngram_range=(1,2)</code> was the right call.</div>",
            unsafe_allow_html=True,
        )
    with insight_cols[2]:
        st.markdown(
            '<div class="insight-box">'
            "<b>Borderline = maximum uncertainty</b> -- the model "
            "correctly assigns ~0.50 probability when sentiment signals "
            "conflict, routing these cases to human review.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# ===== TAB 4 -- PRODUCTION MONITOR =====
with tab4:
    st.markdown("### Is the model still reliable?")

    drift_data = metrics.get("drift")
    if drift_data is None:
        st.warning(
            "Drift summary not found. "
            "Run `python -m monitoring.drift_report` first."
        )
    else:
        drift_detected = drift_data.get("drift_detected", False)
        if drift_detected:
            status_text = "DRIFT DETECTED"
            status_color = "#b43c3c"
        else:
            status_text = "NO DRIFT"
            status_color = "#2ca02c"

        st.markdown(
            '<div class="metric-card" '
            f'style="border-color:{status_color}">'
            f'<p class="value" style="color:{status_color}">'
            f"{status_text}</p></div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                _metric_card(
                    "Reference Samples",
                    f"{drift_data.get('n_reference', 0):,}",
                    "#4a6fa5",
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                _metric_card(
                    "Current Samples",
                    f"{drift_data.get('n_current', 0):,}",
                    "#4a6fa5",
                ),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                _metric_card(
                    "Drift Score",
                    f"{drift_data.get('drift_score', 0):.6f}",
                    status_color,
                ),
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                _metric_card("Threshold", "0.05", "#cdd6f4"),
                unsafe_allow_html=True,
            )

        prob_cols = st.columns(2)
        with prob_cols[0]:
            st.markdown(
                _metric_card(
                    "Mean Prob (Reference)",
                    f"{drift_data.get('mean_prob_reference', 0):.6f}",
                    "#4a6fa5",
                ),
                unsafe_allow_html=True,
            )
        with prob_cols[1]:
            st.markdown(
                _metric_card(
                    "Mean Prob (Current)",
                    f"{drift_data.get('mean_prob_current', 0):.6f}",
                    "#4a6fa5",
                ),
                unsafe_allow_html=True,
            )

        html_path = FIGURES_DIR / "drift_report.html"
        if html_path.exists():
            st.markdown("#### Evidently Drift Report")
            with open(html_path) as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)
        else:
            st.info("Drift HTML report not found at expected path.")

    st.markdown("---")
    st.markdown("#### Confidence Routing Distribution (Test Set)")

    test_results_data = metrics.get("test_results") or {}
    champion_key_rt = test_results_data.get(
        "champion", "lr_tuned_calibrated",
    )
    if champion_key_rt in test_results_data:
        try:
            from src.data.cleaner import clean_data
            from src.data.loader import load_raw_data

            @st.cache_data
            def _get_routing_distribution():
                raw = load_raw_data(config)
                df = clean_data(raw, config)
                test = df[
                    df[config.data.split_column] == config.data.test_label
                ]
                texts = test[config.data.text_column].apply(
                    normalize_text,
                )
                probas = model.predict_proba(texts.tolist())[:, 1]
                auto = int(
                    np.sum((probas > 0.85) | (probas < 0.15)),
                )
                review = int(np.sum(
                    ((probas > 0.60) & (probas <= 0.85))
                    | ((probas >= 0.15) & (probas < 0.40)),
                ))
                escalate = int(
                    np.sum((probas >= 0.40) & (probas <= 0.60)),
                )
                return auto, review, escalate, len(probas)

            auto, review_ct, escalate, total = (
                _get_routing_distribution()
            )
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    _metric_card(
                        "Auto-Classify",
                        f"{auto:,} ({auto / total:.1%})",
                        "#2ca02c",
                    ),
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    _metric_card(
                        "Human Review",
                        f"{review_ct:,} ({review_ct / total:.1%})",
                        "#b08c28",
                    ),
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    _metric_card(
                        "Escalate",
                        f"{escalate:,} ({escalate / total:.1%})",
                        "#b43c3c",
                    ),
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.warning(
                f"Could not compute routing distribution: {exc}",
            )

    explain_cols = st.columns(2)
    with explain_cols[0]:
        st.markdown(
            '<div class="insight-box">'
            "<b>What is drift?</b> Drift occurs when the statistical "
            "properties of incoming data diverge from the training "
            "distribution. This can cause model accuracy to degrade "
            "silently without explicit errors.</div>",
            unsafe_allow_html=True,
        )
    with explain_cols[1]:
        st.markdown(
            '<div class="insight-box">'
            "<b>When to retrain?</b> If drift score exceeds 0.05 "
            "consistently over multiple monitoring windows, or if "
            "business KPIs (e.g., customer complaints) increase, "
            "trigger a retraining pipeline with fresh data.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
