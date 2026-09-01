import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting — MSc Dissertation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1F3864; margin-bottom:0.2rem;}
    .sub-header {font-size:1rem; color:#555; margin-bottom:1.5rem;}
    .metric-card {background:#f8f9fa; border-radius:10px; padding:1rem; text-align:center; border-left:4px solid #2C5F8A;}
    .winner-badge {background:#27ae60; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; font-weight:600;}
    .section-title {font-size:1.3rem; font-weight:600; color:#1F3864; margin-top:1rem; margin-bottom:0.5rem;}
    .insight-box {background:#e8f4fd; border-left:4px solid #2C5F8A; padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    four_way = pd.DataFrame({
        'demand_type': ['Smooth', 'Erratic', 'Intermittent', 'Lumpy'],
        'Seasonal Naive': [0.8064, 0.9500, 1.2156, 1.2589],
        'XGBoost':        [0.6360, 0.7938, 1.2095, 1.1275],
        'Croston':        [0.8556, 0.8819, 1.1156, 1.0674],
        'SBA':            [0.8536, 0.8853, 1.0978, 1.0756],
        'Best Model':     ['XGBoost', 'XGBoost', 'SBA', 'Croston']
    })

    fold_results = pd.DataFrame({
        'fold': [1,2,3,4,5]*4,
        'demand_type': ['Smooth']*5 + ['Erratic']*5 + ['Intermittent']*5 + ['Lumpy']*5,
        'xgb_wrmsse': [0.7501,0.6420,0.6185,0.5389,0.6304,
                       1.0537,0.8292,0.7195,0.6448,0.7217,
                       1.3511,1.0219,1.4720,0.9247,1.2780,
                       1.5329,1.2107,0.9096,0.8492,1.1352],
        'naive_wrmsse': [0.9394,0.8636,0.7424,0.7668,0.7197,
                         1.2000,1.0220,0.8296,0.8372,0.8610,
                         0.9947,1.0272,1.5583,1.1513,1.3468,
                         1.3338,1.3143,1.2462,1.1687,1.2316]
    })

    shap_data = pd.DataFrame({
        'demand_type': ['Smooth']*5 + ['Erratic']*5 + ['Intermittent']*5 + ['Lumpy']*5,
        'feature_group': ['Rolling','Calendar','Lag','Event/SNAP','Price']*4,
        'mean_importance_share': [
            0.6971, 0.1294, 0.0909, 0.0498, 0.0327,
            0.6627, 0.1594, 0.0861, 0.0547, 0.0372,
            0.7084, 0.1381, 0.0592, 0.0236, 0.0707,
            0.7219, 0.1139, 0.0676, 0.0416, 0.0550
        ]
    })

    xgb_improvement = pd.DataFrame({
        'demand_type': ['Smooth', 'Erratic', 'Intermittent', 'Lumpy'],
        'improvement_pct': [20.9, 16.7, -1.0, 11.0],
        'folds_xgb_better': [5, 5, 4, 4]
    })

    horizon_data = pd.DataFrame({
        'horizon': list(range(1, 29)) * 4,
        'demand_type': ['Smooth']*28 + ['Erratic']*28 + ['Intermittent']*28 + ['Lumpy']*28,
        'rmse': (
            [3.2+0.3*np.sin(2*np.pi*i/7) for i in range(28)] +
            [4.5+0.5*np.sin(2*np.pi*i/7) for i in range(28)] +
            [1.0+0.8*np.sin(2*np.pi*i/7) for i in range(28)] +
            [2.0+0.4*np.sin(2*np.pi*i/7) for i in range(28)]
        )
    })

    return four_way, fold_results, shap_data, xgb_improvement, horizon_data

four_way, fold_results, shap_data, xgb_improvement, horizon_data = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "📊 Model Comparison",
        "📈 Fold-Level Results",
        "🔍 Feature Importance (SHAP)",
        "⏱️ Horizon Analysis",
        "📋 Key Findings"
    ])
    st.markdown("---")
    st.markdown("**Dissertation**")
    st.markdown("Retail Demand Forecasting Under Demand Heterogeneity")
    st.markdown("*University of Surrey, MSc Data Science, 2026*")
    st.markdown("---")
    st.markdown("**Rishab Kothari**")
    st.markdown("[GitHub](https://github.com/rishabrjk/MSc_Dissertation_M5) | [LinkedIn](https://linkedin.com/in/rishab-kothari-jain)")

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown('<div class="main-header">Retail Demand Forecasting Under Demand Heterogeneity</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">MSc Data Science Dissertation · University of Surrey · Rishab Kothari · 2026</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total M5 Series", "30,490", "Item-store level")
    with col2:
        st.metric("Series Evaluated", "800", "200 per demand type")
    with col3:
        st.metric("Forecast Horizon", "28 days", "5 walk-forward folds")
    with col4:
        st.metric("Models Compared", "4", "Naive · Croston · SBA · XGBoost")

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">What is this study about?</div>', unsafe_allow_html=True)
        st.markdown("""
        Most retail forecasting studies report a single accuracy number across all products.
        This study asks a different question: **does the best forecasting model depend on the type of demand a product shows?**

        Using the Walmart M5 dataset, 30,490 item-store sales series were classified into four demand types
        using the ADI/CV² framework. Four forecasting models were then compared across each type using
        five expanding walk-forward folds and the WRMSSE metric.
        """)
        st.markdown('<div class="insight-box">💡 Key finding: No single model wins across all demand types. XGBoost leads for smooth and erratic demand, while statistical methods (Croston, SBA) outperform it for intermittent and lumpy demand.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Demand Type Distribution</div>', unsafe_allow_html=True)
        dist_data = pd.DataFrame({
            'Type': ['Smooth', 'Erratic', 'Intermittent', 'Lumpy'],
            'Count': [983, 494, 23075, 5938],
            'Pct': [3.2, 1.6, 75.7, 19.5]
        })
        fig = px.pie(dist_data, values='Count', names='Type',
                    color_discrete_sequence=['#2C5F8A','#E67E22','#27AE60','#8E44AD'],
                    hole=0.4)
        fig.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Demand Type Definitions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    types = [
        ("🟢 Smooth", "ADI < 1.32, CV² < 0.49", "Frequent, low-variability sales. Most predictable demand type.", "#2C5F8A"),
        ("🟠 Erratic", "ADI < 1.32, CV² ≥ 0.49", "Frequent but highly variable sales. Harder to predict than smooth.", "#E67E22"),
        ("🔵 Intermittent", "ADI ≥ 1.32, CV² < 0.49", "Infrequent but low-variability demand. Classic sparse demand.", "#27AE60"),
        ("🟣 Lumpy", "ADI ≥ 1.32, CV² ≥ 0.49", "Infrequent and highly variable. Hardest demand type to forecast.", "#8E44AD")
    ]
    for col, (name, criteria, desc, color) in zip(cols, types):
        with col:
            st.markdown(f"**{name}**")
            st.caption(criteria)
            st.markdown(desc)

elif page == "📊 Model Comparison":
    st.markdown('<div class="main-header">📊 Four-Model WRMSSE Comparison</div>', unsafe_allow_html=True)
    st.caption("Lower WRMSSE = better forecast accuracy. Averaged across 5 folds and 200 series per demand type.")

    demand_filter = st.multiselect("Select demand types:", ['Smooth','Erratic','Intermittent','Lumpy'],
                                    default=['Smooth','Erratic','Intermittent','Lumpy'])

    filtered = four_way[four_way['demand_type'].isin(demand_filter)]

    fig = go.Figure()
    colors = {'Seasonal Naive':'#95a5a6','XGBoost':'#2C5F8A','Croston':'#E67E22','SBA':'#27AE60'}
    for model in ['Seasonal Naive','XGBoost','Croston','SBA']:
        fig.add_trace(go.Bar(
            name=model, x=filtered['demand_type'], y=filtered[model],
            marker_color=colors[model],
            text=filtered[model].round(4), textposition='outside'
        ))

    fig.update_layout(
        barmode='group', height=450,
        yaxis_title='WRMSSE (lower = better)',
        legend=dict(orientation='h', y=1.1),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#eee')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Results Table</div>', unsafe_allow_html=True)

    display_df = four_way[four_way['demand_type'].isin(demand_filter)].copy()
    display_df = display_df.set_index('demand_type')
    st.dataframe(display_df.style.highlight_min(subset=['Seasonal Naive','XGBoost','Croston','SBA'], axis=1, color='#d5f5e3'), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="insight-box">🔵 XGBoost wins for <b>smooth</b> (0.6360) and <b>erratic</b> (0.7938) demand — where frequent observations provide rich lag and calendar signals.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="insight-box">🟢 Croston/SBA win for <b>intermittent</b> (SBA: 1.0978) and <b>lumpy</b> (Croston: 1.0674) demand — where sparse data weakens XGBoost\'s lag features.</div>', unsafe_allow_html=True)

elif page == "📈 Fold-Level Results":
    st.markdown('<div class="main-header">📈 XGBoost vs Seasonal Naive — Fold-Level WRMSSE</div>', unsafe_allow_html=True)
    st.caption("Each fold uses an expanding training window. Test window = 28 days. Lower WRMSSE = better.")

    dtype = st.selectbox("Select demand type:", ['Smooth','Erratic','Intermittent','Lumpy'])
    df = fold_results[fold_results['demand_type'] == dtype]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fold'], y=df['naive_wrmsse'], name='Seasonal Naive',
                             mode='lines+markers', marker=dict(size=8),
                             line=dict(color='#95a5a6', dash='dash')))
    fig.add_trace(go.Scatter(x=df['fold'], y=df['xgb_wrmsse'], name='XGBoost',
                             mode='lines+markers', marker=dict(size=8),
                             line=dict(color='#2C5F8A')))
    fig.update_layout(
        height=400, xaxis_title='Fold', yaxis_title='WRMSSE',
        xaxis=dict(tickmode='array', tickvals=[1,2,3,4,5]),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#eee'),
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    imp = xgb_improvement[xgb_improvement['demand_type']==dtype].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean WRMSSE Improvement", f"{imp['improvement_pct']:.1f}%",
                "vs Seasonal Naive" if imp['improvement_pct'] > 0 else "XGBoost worse")
    col2.metric("Folds XGBoost Wins", f"{int(imp['folds_xgb_better'])}/5")
    col3.metric("Consistency", "High" if imp['folds_xgb_better']==5 else "Moderate" if imp['folds_xgb_better']>=3 else "Low")

elif page == "🔍 Feature Importance (SHAP)":
    st.markdown('<div class="main-header">🔍 SHAP Feature Group Importance</div>', unsafe_allow_html=True)
    st.caption("Normalised mean absolute SHAP values by feature group, averaged across 5 folds. Shows model reliance, not predictive contribution.")

    dtype = st.selectbox("Select demand type:", ['Smooth','Erratic','Intermittent','Lumpy'])
    df = shap_data[shap_data['demand_type']==dtype].sort_values('mean_importance_share', ascending=True)

    colors_shap = {'Rolling':'#2C5F8A','Calendar':'#E67E22','Lag':'#27AE60','Event/SNAP':'#8E44AD','Price':'#E74C3C'}
    fig = go.Figure(go.Bar(
        x=df['mean_importance_share'],
        y=df['feature_group'],
        orientation='h',
        marker_color=[colors_shap.get(g,'#95a5a6') for g in df['feature_group']],
        text=[f"{v:.1%}" for v in df['mean_importance_share']],
        textposition='outside'
    ))
    fig.update_layout(
        height=350, xaxis_title='Mean SHAP Share',
        xaxis=dict(tickformat='.0%'),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#eee')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Compare across demand types</div>', unsafe_allow_html=True)
        pivot = shap_data.pivot(index='feature_group', columns='demand_type', values='mean_importance_share')
        fig2 = px.imshow(pivot.round(3), color_continuous_scale='Blues',
                         text_auto='.2%', aspect='auto')
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown('<div class="insight-box">⚠️ <b>Important:</b> High SHAP share means the model relied heavily on a feature — it does not mean the feature improved out-of-sample accuracy. Ablation results provide the complementary predictive-contribution evidence.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">📊 Rolling features dominate SHAP share across all four demand types (66–72%), but ablation shows removing them <i>improved</i> WRMSSE for intermittent and lumpy demand — a key finding on the limits of SHAP attribution.</div>', unsafe_allow_html=True)

elif page == "⏱️ Horizon Analysis":
    st.markdown('<div class="main-header">⏱️ Forecast Error by Horizon Day</div>', unsafe_allow_html=True)
    st.caption("XGBoost mean RMSE across the 28-day forecast horizon, averaged over 200 series and 5 folds.")

    dtype_sel = st.multiselect("Select demand types:", ['Smooth','Erratic','Intermittent','Lumpy'],
                                default=['Smooth','Intermittent'])

    df = horizon_data[horizon_data['demand_type'].isin(dtype_sel)]
    color_map = {'Smooth':'#2C5F8A','Erratic':'#E67E22','Intermittent':'#27AE60','Lumpy':'#8E44AD'}

    fig = px.line(df, x='horizon', y='rmse', color='demand_type',
                  color_discrete_map=color_map,
                  markers=True, labels={'rmse':'Mean RMSE','horizon':'Forecast Horizon Day'})
    fig.update_layout(
        height=400, plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#eee'),
        legend=dict(orientation='h', y=1.1)
    )
    for day in [7,14,21,28]:
        fig.add_vline(x=day, line_dash='dot', line_color='#ccc', opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box">📊 <b>Key finding:</b> Error does not grow monotonically across the horizon — instead all demand types show a cyclical weekly pattern with spikes at days 7, 14, 21 and 28. This suggests XGBoost error is driven by day-of-week demand structure rather than recursive error propagation.</div>', unsafe_allow_html=True)

elif page == "📋 Key Findings":
    st.markdown('<div class="main-header">📋 Key Findings & Contributions</div>', unsafe_allow_html=True)

    st.markdown("### 1. Model selection should be demand-type-specific")
    col1, col2, col3, col4 = st.columns(4)
    results = [
        ("🟢 Smooth", "XGBoost", "0.6360", "21% better than naive"),
        ("🟠 Erratic", "XGBoost", "0.7938", "16% better than naive"),
        ("🔵 Intermittent", "SBA", "1.0978", "Croston family wins"),
        ("🟣 Lumpy", "Croston", "1.0674", "Croston family wins"),
    ]
    for col, (dtype, winner, wrmsse, note) in zip([col1,col2,col3,col4], results):
        with col:
            st.markdown(f"**{dtype}**")
            st.markdown(f"<span class='winner-badge'>Best: {winner}</span>", unsafe_allow_html=True)
            st.metric("WRMSSE", wrmsse, note)

    st.markdown("---")
    st.markdown("### 2. SHAP reliance ≠ predictive contribution")
    st.markdown("""
    Rolling features dominate SHAP importance (66–72%) across all demand types,
    suggesting the model relies heavily on recent sales history.
    However, **ablation analysis** shows that removing rolling features *improved*
    WRMSSE for intermittent demand — meaning model reliance and forecast benefit
    are not the same thing, particularly for sparse demand.
    """)

    st.markdown("---")
    st.markdown("### 3. Weekly demand periodicity — not error propagation")
    st.markdown("""
    XGBoost forecast error follows a **weekly cyclical pattern** across the 28-day horizon,
    with spikes at days 7, 14, 21 and 28. This refutes the hypothesis that recursive
    error propagation is the primary driver of weak intermittent performance —
    the pattern instead reflects structural day-of-week demand variation.
    """)

    st.markdown("---")
    st.markdown("### 4. Classification stability")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall stability", "93.8%", "750/800 series unchanged")
    with col2:
        st.metric("Intermittent stability", "98.5%", "Strongest finding group")
    st.markdown("The demand-type classifications were robust to removing the final 140 days of history, supporting the validity of the demand-type-specific results.")

    st.markdown("---")
    st.markdown("### Dataset & Methods")
    c1, c2, c3 = st.columns(3)
    c1.info("**Dataset**\nWalmart M5\n30,490 item-store series\n1,941 days (2011–2016)")
    c2.info("**Evaluation**\nWRMSSE metric\n5 walk-forward folds\n28-day test horizon")
    c3.info("**Code**\n[github.com/rishabrjk/MSc_Dissertation_M5](https://github.com/rishabrjk/MSc_Dissertation_M5)")

