# theme.py — Visual BK Engenharia (idêntico ao ERP)
"""
Paleta de cores, CSS customizado, KPI cards e helpers visuais.
Padrão dark glassmorphism com gradiente BK azul.
"""
import streamlit as st

# ═══════════════════════════════════════════
# PALETA BK
# ═══════════════════════════════════════════
BK_BLUE       = "#1565C0"
BK_BLUE_LIGHT = "#42A5F5"
BK_TEAL       = "#00897B"
BK_GREEN      = "#43A047"
BK_ORANGE     = "#FB8C00"
BK_RED        = "#E53935"
BK_PURPLE     = "#7B1FA2"
BK_GRAY       = "#546E7A"
BK_DARK       = "#0D1B2A"
BK_CARD       = "#FFFFFF"

BK_COLORS = [BK_BLUE, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE, BK_GRAY, BK_BLUE_LIGHT]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Segoe UI, Arial", size=12, color=BK_DARK),
    margin=dict(l=30, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Segoe UI"),
)


def apply_bk_theme():
    """Injeta CSS BK em qualquer página Streamlit."""
    st.markdown("""
    <style>
    /* ═══ FONTES ═══ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%);
    }
    [data-testid="stSidebar"] * { color: #E0E0E0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label { font-weight: 500; }

    /* ═══ SIDEBAR NAV — botões estilizados sem duplo texto ═══ */
    [data-testid="stSidebar"] [data-testid="stButton"] {
        margin: 1px 0 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background: rgba(255,255,255,0.03) !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 0 8px 8px 0 !important;
        box-shadow: none !important;
        color: #B0BEC5 !important;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
        text-align: left !important;
        padding: 9px 14px !important;
        width: 100% !important;
        transition: all 0.15s ease !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        background: rgba(255,255,255,0.07) !important;
        border-left-color: #5C9BD6 !important;
        color: #E0E0E0 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    /* ═══ BOTÕES GRADIENTE (fora da sidebar) ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(21,101,192,0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(21,101,192,0.5);
    }

    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 500;
        color: #546E7A;
    }
    .stTabs [aria-selected="true"] {
        color: #1565C0 !important;
        border-bottom: 3px solid #1565C0;
        font-weight: 600;
    }

    /* ═══ DATA EDITOR (tabelas editáveis) ═══ */
    [data-testid="stDataFrame"],
    [data-testid="data-grid-canvas"] {
        border-radius: 8px;
        border: 1px solid #E0E0E0;
    }

    /* ═══ KPI CARDS ═══ */
    .bk-kpi-row {
        display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0 16px 0;
    }
    .bk-kpi {
        flex: 1; min-width: 140px;
        background: #fff;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #1565C0;
        transition: transform 0.15s;
    }
    .bk-kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
    .bk-kpi-value { font-size: 1.5rem; font-weight: 700; color: #0D1B2A; }
    .bk-kpi-label { font-size: 0.78rem; color: #546E7A; margin-top: 2px; }

    .bk-kpi-blue   { color: #1565C0 !important; }
    .bk-kpi-green  { color: #43A047 !important; }
    .bk-kpi-teal   { color: #00897B !important; }
    .bk-kpi-orange { color: #FB8C00 !important; }
    .bk-kpi-red    { color: #E53935 !important; }
    .bk-kpi-gray   { color: #546E7A !important; }

    /* ═══ SECTION TITLE ═══ */
    .bk-section {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0D1B2A;
        border-left: 4px solid #1565C0;
        padding: 6px 12px;
        margin: 18px 0 10px 0;
        background: #F5F8FC;
        border-radius: 0 6px 6px 0;
    }

    /* ═══ TOOLTIP nos inputs ═══ */
    .stTooltipIcon { color: #1565C0 !important; }

    /* ═══ INPUTS — texto escuro legível em fundos claros ═══ */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stDateInput input {
        color: #1a1a2e !important;
        font-weight: 500 !important;
    }
    /* Selectbox / dropdown: texto escuro */
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div[class*="value"],
    .stSelectbox [data-baseweb="select"] input,
    .stMultiSelect [data-baseweb="select"] span,
    [data-baseweb="select"] .css-1dimb5e-singleValue,
    [data-baseweb="popover"] li {
        color: #1a1a2e !important;
        font-weight: 500 !important;
    }
    /* Placeholder mais visível */
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #546E7A !important;
        opacity: 0.8 !important;
    }
    /* Labels nos inputs (fora da sidebar) */
    .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stDateInput label,
    .stTextArea label, .stMultiSelect label {
        color: #0D1B2A !important;
        font-weight: 500 !important;
    }

    /* ═══ HEADER GRADIENTE ═══ */
    .bk-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(13,27,42,0.3);
    }
    .bk-header h1 { color: white; margin: 0; font-size: 1.6rem; }
    .bk-header p  { color: #90CAF9; margin: 4px 0 0 0; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)


def bk_header(title: str, subtitle: str = ""):
    """Header gradiente padrão BK."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="bk-header"><h1>⚡ {title}</h1>{sub}</div>', unsafe_allow_html=True)


def bk_section(title: str):
    """Título de seção com barra lateral azul."""
    st.markdown(f'<div class="bk-section">{title}</div>', unsafe_allow_html=True)


def bk_kpi(label: str, value: str, color: str = "blue") -> str:
    return f'''<div class="bk-kpi">
        <div class="bk-kpi-value bk-kpi-{color}">{value}</div>
        <div class="bk-kpi-label">{label}</div>
    </div>'''


def bk_kpi_row(cards: list):
    """cards = [(label, value, color), ...]"""
    html = '<div class="bk-kpi-row">'
    for label, value, color in cards:
        html += bk_kpi(label, str(value), color)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
