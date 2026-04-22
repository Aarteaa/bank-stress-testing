import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Bank Stress Testing | RBI-Aligned",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.rbi-header {
    background: linear-gradient(135deg,#1a2744,#0d1b35);
    border-left: 4px solid #e8b84b;
    padding: 18px 22px; border-radius: 4px; margin-bottom: 20px;
}
.rbi-header h1 { color:#e8b84b; font-size:1.5rem; margin:0; font-weight:700; }
.rbi-header p  { color:#9ba3b8; font-size:0.83rem; margin:5px 0 0; }
.metric-card {
    background:#1a1f2e; border:1px solid #2a3044;
    border-radius:6px; padding:14px 18px; text-align:center;
}
.metric-card .label { color:#9ba3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:.05em; }
.metric-card .value { color:#e8e8e8; font-size:1.5rem; font-weight:700; font-family:'IBM Plex Mono',monospace; margin:4px 0; }
.metric-card .delta { font-size:0.76rem; font-family:'IBM Plex Mono',monospace; }
.bad  { color:#e05252; } .ok { color:#52b788; } .warn { color:#e8b84b; }
.pass-badge { background:#1a3a2a; color:#52b788; border:1px solid #52b788; padding:4px 10px; border-radius:20px; font-size:0.76rem; font-weight:600; }
.fail-badge { background:#3a1a1a; color:#e05252; border:1px solid #e05252; padding:4px 10px; border-radius:20px; font-size:0.76rem; font-weight:600; }
.warn-badge { background:#3a301a; color:#e8b84b; border:1px solid #e8b84b; padding:4px 10px; border-radius:20px; font-size:0.76rem; font-weight:600; }
.section-title { color:#e8b84b; font-size:0.68rem; text-transform:uppercase; letter-spacing:.15em; font-weight:600; border-bottom:1px solid #2a3044; padding-bottom:7px; margin:26px 0 14px; }
.insight-box { background:#141824; border-left:3px solid #e8b84b; padding:11px 15px; border-radius:0 4px 4px 0; margin:7px 0; font-size:0.83rem; color:#c8cdd8; }
.data-note { background:#1a2030; border:1px solid #2a3550; border-radius:4px; padding:9px 13px; font-size:0.76rem; color:#7a8399; margin-top:8px; }
div[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e2433; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme defaults ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0f1117',
    plot_bgcolor='#1a1f2e',
    font=dict(family='IBM Plex Sans', color='#9ba3b8', size=11),
    title_font=dict(color='#e0e4f0', size=13),
    margin=dict(l=40, r=20, t=45, b=40),
    xaxis=dict(gridcolor='#2a3044', linecolor='#2a3044', zerolinecolor='#2a3044'),
    yaxis=dict(gridcolor='#2a3044', linecolor='#2a3044', zerolinecolor='#2a3044'),
    legend=dict(bgcolor='#1a1f2e', bordercolor='#2a3044', borderwidth=1, font=dict(size=10)),
)
SC_COLORS = ['#52b788', '#e8b84b', '#e07b52', '#e05252']

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rbi-header">
  <h1>🏦 India Bank Stress Testing Framework</h1>
  <p>RBI-Aligned Macro Scenarios · Basel III / SFB Compliance · Monte Carlo VaR/CVaR · SHAP Feature Attribution</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🏛️ Bank Portfolio Parameters")
bank_type = st.sidebar.selectbox("Bank Type", [
    "Small Finance Bank", "Regional Rural Bank",
    "Urban Co-operative Bank", "Public Sector Bank"
])
portfolio_size = st.sidebar.number_input("Loan Portfolio Size (₹ Crore)", 100, 500000, 8500, 100)
current_gnpa   = st.sidebar.number_input("Current GNPA Ratio (%)", 0.5, 20.0, 4.2, 0.1,
    help="Gross Non-Performing Assets ratio per RBI classification")
current_car    = st.sidebar.number_input("Current CRAR (%)", 9.0, 25.0, 15.5, 0.5,
    help="Capital to Risk-weighted Assets Ratio per RBI/Basel III")
tier1_ratio    = st.sidebar.number_input("Tier 1 Capital Ratio (%)", 6.0, 20.0, 13.2, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Stress Scenario")
scenario = st.sidebar.selectbox("Select Macro Scenario", [
    "Baseline",
    "Moderate Stress (RBI Adverse)",
    "Severe Stress (RBI Severely Adverse)",
    "Tail Risk / Crisis"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎲 Monte Carlo Settings")
n_simulations   = st.sidebar.slider("Simulations", 500, 5000, 2000, 500)
confidence_level= st.sidebar.selectbox("Confidence Level", ["95%", "99%", "99.5%"])
conf_val        = {"95%": 0.95, "99%": 0.99, "99.5%": 0.995}[confidence_level]

# ── Data & Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def build_and_train():
    np.random.seed(42)
    n = 600

    gdp_growth      = np.random.normal(6.5,  2.8, n)
    unemployment    = np.random.normal(7.0,  2.5, n)
    repo_rate       = np.random.normal(6.0,  1.5, n)
    cpi_inflation   = np.random.normal(5.5,  1.8, n)
    iip_growth      = np.random.normal(4.0,  6.0, n)
    credit_growth   = np.random.normal(12.0, 6.0, n)
    nifty_return    = np.random.normal(12.0, 22.0, n)
    usd_inr_change  = np.random.normal(2.5,  4.5, n)
    wpi_inflation   = np.random.normal(4.0,  3.5, n)
    msme_stress_idx = np.clip(np.random.normal(0.3, 0.2, n), 0, 1)

    gnpa = (
        5.0
        - 0.35 * gdp_growth
        + 0.30 * unemployment
        + 0.15 * repo_rate
        - 0.05 * iip_growth
        + 0.10 * np.maximum(0, credit_growth - 18)
        + 0.20 * usd_inr_change
        + 1.50 * msme_stress_idx
        + 0.08 * np.maximum(0, cpi_inflation - 6)
        + np.random.normal(0, 0.6, n)
    )
    gnpa = np.clip(gnpa, 1.0, 30.0)

    df = pd.DataFrame({
        'gdp_growth': gdp_growth, 'unemployment': unemployment,
        'repo_rate': repo_rate, 'cpi_inflation': cpi_inflation,
        'iip_growth': iip_growth, 'credit_growth': credit_growth,
        'nifty_return': nifty_return, 'usd_inr_change': usd_inr_change,
        'wpi_inflation': wpi_inflation, 'msme_stress_idx': msme_stress_idx,
        'gnpa_ratio': gnpa
    })

    X = df.drop('gnpa_ratio', axis=1)
    y = df['gnpa_ratio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
    gb.fit(X_tr, y_train)

    rf_p = rf.predict(X_te);  gb_p = gb.predict(X_te)
    metrics = {
        'RF': dict(RMSE=np.sqrt(mean_squared_error(y_test,rf_p)), MAE=mean_absolute_error(y_test,rf_p), R2=r2_score(y_test,rf_p)),
        'GB': dict(RMSE=np.sqrt(mean_squared_error(y_test,gb_p)), MAE=mean_absolute_error(y_test,gb_p), R2=r2_score(y_test,gb_p)),
    }
    return rf, gb, scaler, X.columns.tolist(), df, metrics

with st.spinner("Training models on India-calibrated data..."):
    rf_model, gb_model, scaler, feature_names, hist_data, model_metrics = build_and_train()

st.success("✅ Models trained on RBI-calibrated synthetic data (600 quarters)")

# ── Scenarios ──────────────────────────────────────────────────────────────────
SCENARIOS = {
    "Baseline": {
        'gdp_growth':6.8,'unemployment':7.0,'repo_rate':6.5,'cpi_inflation':4.5,
        'iip_growth':5.0,'credit_growth':13.0,'nifty_return':12.0,
        'usd_inr_change':2.0,'wpi_inflation':4.0,'msme_stress_idx':0.25,
        'desc':'Normal growth. RBI FSR 2024 baseline assumptions.'
    },
    "Moderate Stress (RBI Adverse)": {
        'gdp_growth':4.0,'unemployment':9.5,'repo_rate':7.5,'cpi_inflation':6.5,
        'iip_growth':1.0,'credit_growth':6.0,'nifty_return':-8.0,
        'usd_inr_change':6.0,'wpi_inflation':7.0,'msme_stress_idx':0.55,
        'desc':'Adverse scenario per RBI FSR. Growth slowdown + rate pressure + INR stress.'
    },
    "Severe Stress (RBI Severely Adverse)": {
        'gdp_growth':1.5,'unemployment':13.0,'repo_rate':8.0,'cpi_inflation':8.5,
        'iip_growth':-4.0,'credit_growth':-2.0,'nifty_return':-28.0,
        'usd_inr_change':12.0,'wpi_inflation':10.0,'msme_stress_idx':0.80,
        'desc':'Severely adverse. IL&FS + COVID supply shock combined stress level.'
    },
    "Tail Risk / Crisis": {
        'gdp_growth':-2.0,'unemployment':16.0,'repo_rate':9.0,'cpi_inflation':10.0,
        'iip_growth':-10.0,'credit_growth':-8.0,'nifty_return':-45.0,
        'usd_inr_change':18.0,'wpi_inflation':13.0,'msme_stress_idx':0.95,
        'desc':'Tail risk. 2008-level global shock + domestic credit freeze.'
    }
}

LABEL_MAP = {
    'gdp_growth':'GDP Growth Rate','unemployment':'Unemployment (CMIE)',
    'repo_rate':'RBI Repo Rate','cpi_inflation':'CPI Inflation',
    'iip_growth':'IIP Growth','credit_growth':'Bank Credit Growth',
    'nifty_return':'Nifty 50 Returns','usd_inr_change':'USD/INR Depreciation',
    'wpi_inflation':'WPI Inflation','msme_stress_idx':'MSME Stress Index',
}

# ── Helper functions ───────────────────────────────────────────────────────────
def predict_gnpa(params, noise_std=0.0):
    p = {k: v for k, v in params.items() if k != 'desc'}
    df_s = pd.DataFrame([p])
    if noise_std > 0:
        df_s += np.random.normal(0, noise_std, df_s.shape)
    scaled = scaler.transform(df_s)
    return max((rf_model.predict(scaled)[0] + gb_model.predict(scaled)[0]) / 2, 1.0)

def capital_impact(gnpa_pct, portfolio, base_gnpa, crar, tier1):
    lgd = 0.45
    el  = (portfolio * gnpa_pct / 100) * lgd
    prov_shortfall = max(0, el * 0.70 - portfolio * 0.03)
    npl_mult   = gnpa_pct / base_gnpa
    rwa        = portfolio * 0.85 * (1 + (npl_mult - 1) * 0.25)
    cap_eroded = el + prov_shortfall * 0.5
    cur_cap    = portfolio * crar / 100
    str_crar   = ((cur_cap - cap_eroded) / rwa) * 100
    tier1_eros = cap_eroded / cur_cap
    str_tier1  = tier1 * (1 - tier1_eros * 0.8)
    return dict(gnpa_pct=gnpa_pct, el=el, prov=prov_shortfall,
                cap_eroded=cap_eroded, str_crar=str_crar, str_tier1=str_tier1, rwa=rwa)

@st.cache_data
def run_mc(scenario_key, n, portfolio, base_gnpa, crar, tier1, conf):
    p = SCENARIOS[scenario_key]
    gnpas, losses, crars = [], [], []
    for _ in range(n):
        g = predict_gnpa(p, noise_std=0.8)
        r = capital_impact(g, portfolio, base_gnpa, crar, tier1)
        gnpas.append(r['gnpa_pct']); losses.append(r['el']); crars.append(r['str_crar'])
    var_g  = np.percentile(gnpas, conf*100)
    var_l  = np.percentile(losses, conf*100)
    cvar_l = np.mean([l for l in losses if l >= var_l])
    min_c  = np.percentile(crars, (1-conf)*100)
    return dict(gnpas=gnpas, losses=losses, crars=crars,
                var_g=var_g, var_l=var_l, cvar_l=cvar_l, min_c=min_c,
                prob_fail=np.mean([c<9 for c in crars]),
                mean_g=np.mean(gnpas), mean_l=np.mean(losses), mean_c=np.mean(crars))

def shap_approx(params, baseline_params):
    p = {k:v for k,v in params.items() if k!='desc'}
    b = {k:v for k,v in baseline_params.items() if k!='desc'}
    base_g = predict_gnpa(b)
    contribs = {}
    for feat in feature_names:
        m = dict(b); m[feat] = p[feat]
        contribs[feat] = predict_gnpa(m) - base_g
    return contribs, base_g

# ── Run calculations ───────────────────────────────────────────────────────────
scen_p   = SCENARIOS[scenario]
pt_gnpa  = predict_gnpa(scen_p)
pt_res   = capital_impact(pt_gnpa, portfolio_size, current_gnpa, current_car, tier1_ratio)
mc       = run_mc(scenario, n_simulations, portfolio_size, current_gnpa, current_car, tier1_ratio, conf_val)
shap_c, base_g = shap_approx(scen_p, SCENARIOS["Baseline"])

all_res = {}
for sn, sp in SCENARIOS.items():
    g = predict_gnpa(sp)
    all_res[sn] = capital_impact(g, portfolio_size, current_gnpa, current_car, tier1_ratio)

# ── Scenario description ───────────────────────────────────────────────────────
st.markdown(f'<div class="insight-box">📋 <b>{scenario}</b> — {scen_p["desc"]}</div>', unsafe_allow_html=True)

# ── Key metrics ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Point Estimate Results</div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)

gd = pt_gnpa - current_gnpa
cd = pt_res['str_crar'] - current_car
lp = pt_res['el'] / portfolio_size * 100
fp = mc['prob_fail'] * 100

def delta_cls(v, bad=3, warn=1): return "bad" if abs(v)>=bad else "warn" if abs(v)>=warn else "ok"

with c1:
    st.markdown(f"""<div class="metric-card"><div class="label">Stressed GNPA</div>
    <div class="value">{pt_gnpa:.1f}%</div>
    <div class="delta {delta_cls(gd)}">+{gd:.1f}% vs current</div></div>""", unsafe_allow_html=True)
with c2:
    sign = "+" if cd>0 else ""
    st.markdown(f"""<div class="metric-card"><div class="label">Stressed CRAR</div>
    <div class="value">{pt_res['str_crar']:.1f}%</div>
    <div class="delta {'ok' if cd>0 else 'bad'}">{sign}{cd:.1f}% vs current</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="label">Expected Credit Loss</div>
    <div class="value">₹{pt_res['el']:.0f}Cr</div>
    <div class="delta bad">{lp:.1f}% of portfolio</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="label">VaR GNPA ({confidence_level})</div>
    <div class="value">{mc['var_g']:.1f}%</div>
    <div class="delta bad">CVaR Loss ₹{mc['cvar_l']:.0f}Cr</div></div>""", unsafe_allow_html=True)
with c5:
    cls5 = "bad" if fp>10 else "warn" if fp>2 else "ok"
    st.markdown(f"""<div class="metric-card"><div class="label">P(CRAR &lt; 9%)</div>
    <div class="value">{fp:.1f}%</div>
    <div class="delta {cls5}">{n_simulations:,} simulations</div></div>""", unsafe_allow_html=True)

# ── Regulatory compliance ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">RBI / Basel III Regulatory Compliance</div>', unsafe_allow_html=True)
checks = [
    ("CRAR ≥ 9% (RBI Min)",        pt_res['str_crar'] >= 9.0),
    ("CRAR ≥ 15% (SFB Floor)",     pt_res['str_crar'] >= 15.0),
    ("Tier 1 ≥ 7.5%",              pt_res['str_tier1'] >= 7.5),
    ("GNPA < 10% (Supervisory)",   pt_gnpa < 10.0),
    (f"MC CRAR > 9% ({confidence_level})", mc['min_c'] >= 9.0),
]
cols = st.columns(len(checks))
for col,(label,passed) in zip(cols,checks):
    cls = "pass-badge" if passed else "fail-badge"
    icon = "✅" if passed else "❌"
    col.markdown(f"""<div style="text-align:center;padding:8px">
        <span class="{cls}">{icon} {'PASS' if passed else 'FAIL'}</span>
        <div style="font-size:.7rem;color:#7a8399;margin-top:5px">{label}</div>
    </div>""", unsafe_allow_html=True)

# ── Chart 1: Scenario comparison (Plotly) ─────────────────────────────────────
st.markdown('<div class="section-title">Scenario Analysis — All Stress Scenarios</div>', unsafe_allow_html=True)

s_short   = ["Baseline","Moderate","Severe","Tail Risk"]
gnpa_vals = [all_res[s]['gnpa_pct'] for s in SCENARIOS]
crar_vals = [all_res[s]['str_crar'] for s in SCENARIOS]
loss_vals = [all_res[s]['el'] for s in SCENARIOS]
eros_vals = [all_res[s]['cap_eroded'] for s in SCENARIOS]

fig1 = make_subplots(rows=1, cols=4, subplot_titles=[
    "Stressed GNPA (%)", "Stressed CRAR (%)", "Expected Loss (₹Cr)", "Capital Erosion (₹Cr)"
])
for i,(vals,title) in enumerate([(gnpa_vals,''),(crar_vals,''),(loss_vals,''),(eros_vals,'')],1):
    fig1.add_trace(go.Bar(x=s_short, y=vals, marker_color=SC_COLORS,
                          text=[f"{v:.1f}" for v in vals], textposition='outside',
                          textfont=dict(color='#e0e4f0',size=10), showlegend=False), row=1, col=i)

# Reference lines
fig1.add_hline(y=current_gnpa, line_dash="dash", line_color="#7ab8f5", row=1, col=1, annotation_text=f"Current {current_gnpa}%")
fig1.add_hline(y=10, line_dash="dot", line_color="#e05252", row=1, col=1, annotation_text="Threshold 10%")
fig1.add_hline(y=9.0, line_dash="dot", line_color="#e05252", row=1, col=2, annotation_text="Min 9%")
fig1.add_hline(y=15.0, line_dash="dash", line_color="#e8b84b", row=1, col=2, annotation_text="SFB 15%")
fig1.add_hline(y=current_car, line_dash="dash", line_color="#7ab8f5", row=1, col=2, annotation_text=f"Current {current_car}%")

fig1.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False)
fig1.update_layout(annotations=[dict(font=dict(size=11,color='#e0e4f0')) for _ in fig1.layout.annotations])
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Monte Carlo distributions (Plotly) ────────────────────────────────
st.markdown('<div class="section-title">Monte Carlo Simulation — Loss Distribution</div>', unsafe_allow_html=True)

fig2 = make_subplots(rows=1, cols=3, subplot_titles=[
    "GNPA Distribution", "Expected Loss Distribution", "Stressed CRAR Distribution"
])
fig2.add_trace(go.Histogram(x=mc['gnpas'], nbinsx=50, marker_color='#52b788', opacity=0.75, name='GNPA',showlegend=False), row=1,col=1)
fig2.add_vline(x=mc['var_g'], line_dash="dash", line_color="#e8b84b", row=1, col=1, annotation_text=f"VaR {mc['var_g']:.1f}%")
fig2.add_vline(x=pt_gnpa,     line_dash="solid", line_color="#7ab8f5", row=1, col=1, annotation_text=f"Point {pt_gnpa:.1f}%")

fig2.add_trace(go.Histogram(x=mc['losses'], nbinsx=50, marker_color='#e8b84b', opacity=0.75, name='Loss',showlegend=False), row=1,col=2)
fig2.add_vline(x=mc['var_l'],  line_dash="dash", line_color="#e05252", row=1, col=2, annotation_text=f"VaR ₹{mc['var_l']:.0f}Cr")
fig2.add_vline(x=mc['cvar_l'], line_dash="dot",  line_color="#e07b52", row=1, col=2, annotation_text=f"CVaR ₹{mc['cvar_l']:.0f}Cr")

fig2.add_trace(go.Histogram(x=mc['crars'], nbinsx=50, marker_color='#7ab8f5', opacity=0.75, name='CRAR',showlegend=False), row=1,col=3)
fig2.add_vline(x=9.0,        line_dash="dot",  line_color="#e05252", row=1, col=3, annotation_text="RBI Min 9%")
fig2.add_vline(x=mc['min_c'],line_dash="dash", line_color="#e8b84b", row=1, col=3, annotation_text=f"Floor {mc['min_c']:.1f}%")

fig2.update_layout(**PLOTLY_LAYOUT, height=340)
st.plotly_chart(fig2, use_container_width=True)

# ── Chart 3: SHAP waterfall (Plotly) ──────────────────────────────────────────
st.markdown('<div class="section-title">SHAP-Style Feature Attribution — What Drives GNPA Change?</div>', unsafe_allow_html=True)

shap_df = pd.DataFrame({
    'Feature': [LABEL_MAP.get(k,k) for k in shap_c],
    'Value':   list(shap_c.values())
}).sort_values('Value', key=abs)

colors = ['#e05252' if v>0 else '#52b788' for v in shap_df['Value']]
fig3 = go.Figure(go.Bar(
    x=shap_df['Value'], y=shap_df['Feature'],
    orientation='h',
    marker_color=colors,
    text=[f"{v:+.2f}%" for v in shap_df['Value']],
    textposition='outside',
    textfont=dict(color='#e0e4f0', size=10),
))
fig3.add_vline(x=0, line_color='#9ba3b8', line_width=1)
fig3.update_layout(
    **PLOTLY_LAYOUT,
    height=380,
    title=f"Feature Attribution vs Baseline — {scenario}",
    xaxis_title="Contribution to GNPA change (%)",
    yaxis_title="",
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("""<div class="data-note">
🔍 <b>How to read:</b> Red bars increase GNPA (more risk); green bars reduce it (protective). 
MSME Stress Index and Unemployment are typically the top drivers for Small Finance Banks — 
consistent with RBI FSR empirical findings on SFB vulnerability.
</div>""", unsafe_allow_html=True)

# ── Chart 4: Sensitivity heatmap ──────────────────────────────────────────────
st.markdown('<div class="section-title">Sensitivity Analysis — GDP × Unemployment Shock Grid</div>', unsafe_allow_html=True)

gdp_vals  = [-5,-3,-1,0,2,4,6,8]
unemp_vals= [5,7,9,11,13,15]
base_p    = {k:v for k,v in SCENARIOS['Baseline'].items() if k!='desc'}
z = []
for u in unemp_vals:
    row = []
    for g in gdp_vals:
        p2 = dict(base_p); p2['gdp_growth']=g; p2['unemployment']=u
        row.append(round(predict_gnpa(p2), 1))
    z.append(row)

fig4 = go.Figure(go.Heatmap(
    z=z,
    x=[f"GDP {g}%" for g in gdp_vals],
    y=[f"Unemp {u}%" for u in unemp_vals],
    colorscale=[[0,'#1a3a2a'],[0.33,'#3a301a'],[0.66,'#3a2010'],[1,'#3a1a1a']],
    text=[[f"{v}%" for v in row] for row in z],
    texttemplate="%{text}",
    textfont=dict(size=11, color='#e0e4f0'),
    colorbar=dict(title=dict(text="GNPA %", font=dict(color='#9ba3b8')), tickfont=dict(color='#9ba3b8')),
    showscale=True,
))
fig4.update_layout(**PLOTLY_LAYOUT, height=300,
    title="Predicted GNPA (%) — GDP Growth vs Unemployment (all other vars at Baseline)",
    xaxis=dict(side='bottom'))
st.plotly_chart(fig4, use_container_width=True)

# ── Model performance ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("**Random Forest**")
    st.dataframe(pd.DataFrame({'Metric':['RMSE','MAE','R²'],
        'Value':[f"{model_metrics['RF']['RMSE']:.3f}",f"{model_metrics['RF']['MAE']:.3f}",f"{model_metrics['RF']['R2']:.3f}"]}),
        hide_index=True, use_container_width=True)
with mc2:
    st.markdown("**Gradient Boosting**")
    st.dataframe(pd.DataFrame({'Metric':['RMSE','MAE','R²'],
        'Value':[f"{model_metrics['GB']['RMSE']:.3f}",f"{model_metrics['GB']['MAE']:.3f}",f"{model_metrics['GB']['R2']:.3f}"]}),
        hide_index=True, use_container_width=True)

# ── Recommendations ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">RBI-Aligned Capital Planning Recommendations</div>', unsafe_allow_html=True)
buf = max(0, (15.0 - pt_res['str_crar']) * portfolio_size * 0.85 / 100)
if pt_res['str_crar'] < 9.0:
    st.error(f"🚨 **CRITICAL**: CRAR falls below RBI minimum (9%). Immediate capital raise of ~₹{buf:.0f} Cr required.")
elif pt_res['str_crar'] < 15.0:
    st.warning(f"⚠️ **WARNING**: CRAR falls below SFB floor (15%). Capital infusion of ~₹{buf:.0f} Cr recommended.")
else:
    st.success("✅ **ADEQUATE**: Capital position holds above SFB minimum (15%) under this scenario.")

r1, r2 = st.columns(2)
with r1:
    st.markdown(f"""<div class="insight-box">
    💼 <b>Capital Actions:</b><br>
    • Maintain GNPA provision coverage ≥ 70% (RBI guidance)<br>
    • Recommended CRAR target: {max(pt_res['str_crar']+3, 18):.1f}%<br>
    • Additional Tier 1 via AT1 bonds: ₹{buf:.0f} Cr<br>
    • Review dividend policy under stressed scenario
    </div>""", unsafe_allow_html=True)
with r2:
    mod_gnpa = all_res['Moderate Stress (RBI Adverse)']['gnpa_pct']
    st.markdown(f"""<div class="insight-box">
    📡 <b>Early Warning Triggers (RBI PCA Framework):</b><br>
    • Alert threshold: GNPA exceeds {mod_gnpa:.1f}% (Moderate Stress level)<br>
    • PCA trigger: GNPA > 10%, CRAR < 10.25%<br>
    • Enhanced MSME monitoring — top GNPA driver for SFBs<br>
    • Monitor USD/INR — impacts import-linked borrowers
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div class="data-note">
<b>Methodology Note:</b> Synthetic data calibrated to Indian banking system using RBI DBIE historical ranges (FY2005–FY2024). 
Stress scenarios aligned with RBI FSR stress testing methodology. Capital impact model uses Basel III IRB parameters 
per RBI guidelines (DBOD.No.BP.BC.96/21.06.201/2011-12). For production, replace with actual DBIE time series via RBI API.<br><br>
Built by <b>Aarti Navale</b> · <a href="https://github.com/Aarteaa/bank-stress-testing" style="color:#e8b84b">GitHub</a>
</div>""", unsafe_allow_html=True)
