import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Bank Stress Testing Model",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Stress Testing Model - ML Implementation")
st.markdown("""
This interactive dashboard demonstrates machine learning-based stress testing for small finance banks.
Adjust parameters and scenarios to see how the portfolio performs under different economic conditions.
""")

# Sidebar for inputs
st.sidebar.header("📊 Bank Parameters")
portfolio_size = st.sidebar.number_input("Portfolio Size ($M)", min_value=10, max_value=1000, value=100)
current_npl = st.sidebar.number_input("Current NPL Ratio (%)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
capital_ratio = st.sidebar.number_input("Current Capital Ratio (%)", min_value=8.0, max_value=20.0, value=12.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Stress Scenario")

scenario = st.sidebar.selectbox(
    "Select Scenario",
    ["Baseline", "Moderate Stress", "Severe Recession", "Credit Crisis"]
)

# Cache the model training
@st.cache_resource
def train_models():
    """Train ML models and cache them"""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    gdp_growth = np.random.normal(2.0, 2.5, n_samples)
    unemployment = np.random.normal(6.0, 2.0, n_samples)
    interest_rate = np.random.normal(4.0, 1.5, n_samples)
    inflation = np.random.normal(3.0, 1.0, n_samples)
    property_prices = np.random.normal(5.0, 8.0, n_samples)
    credit_spread = np.random.normal(2.0, 1.0, n_samples)
    liquidity_index = np.random.normal(50, 15, n_samples)
    loan_growth = np.random.normal(8.0, 5.0, n_samples)
    deposit_growth = np.random.normal(6.0, 4.0, n_samples)
    
    npl_ratio = (
        3.0 - 0.3 * gdp_growth + 0.4 * unemployment + 0.2 * interest_rate 
        - 0.1 * property_prices + 0.3 * credit_spread
        + np.random.normal(0, 0.5, n_samples)
    )
    npl_ratio = np.clip(npl_ratio, 0.5, 25.0)
    
    data = pd.DataFrame({
        'gdp_growth': gdp_growth, 'unemployment': unemployment,
        'interest_rate': interest_rate, 'inflation': inflation,
        'property_prices': property_prices, 'credit_spread': credit_spread,
        'liquidity_index': liquidity_index, 'loan_growth': loan_growth,
        'deposit_growth': deposit_growth, 'npl_ratio': npl_ratio
    })
    
    X = data.drop('npl_ratio', axis=1)
    y = data['npl_ratio']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    return rf_model, gb_model, scaler, X.columns

# Train models
with st.spinner("Training ML models..."):
    rf_model, gb_model, scaler, feature_names = train_models()

st.success("✅ Models trained successfully!")

# Define scenarios
scenarios = {
    "Baseline": {
        'gdp_growth': 2.5, 'unemployment': 5.0, 'interest_rate': 4.0,
        'inflation': 3.0, 'property_prices': 3.0, 'credit_spread': 1.5,
        'liquidity_index': 60, 'loan_growth': 8.0, 'deposit_growth': 6.0
    },
    "Moderate Stress": {
        'gdp_growth': -1.0, 'unemployment': 8.0, 'interest_rate': 6.0,
        'inflation': 4.5, 'property_prices': -10.0, 'credit_spread': 3.0,
        'liquidity_index': 40, 'loan_growth': 2.0, 'deposit_growth': 1.0
    },
    "Severe Recession": {
        'gdp_growth': -3.5, 'unemployment': 12.0, 'interest_rate': 5.5,
        'inflation': 2.0, 'property_prices': -25.0, 'credit_spread': 5.0,
        'liquidity_index': 25, 'loan_growth': -5.0, 'deposit_growth': -3.0
    },
    "Credit Crisis": {
        'gdp_growth': -2.5, 'unemployment': 10.0, 'interest_rate': 4.5,
        'inflation': 1.5, 'property_prices': -20.0, 'credit_spread': 6.0,
        'liquidity_index': 20, 'loan_growth': -8.0, 'deposit_growth': -5.0
    }
}

# Get selected scenario
scenario_params = scenarios[scenario]

# Predict stress impact
def predict_impact(params, model, scaler, portfolio, current_npl, capital):
    scenario_df = pd.DataFrame([params])
    scenario_scaled = scaler.transform(scenario_df)
    predicted_npl = max(model.predict(scenario_scaled)[0], 0.5)
    
    expected_loss = (portfolio * predicted_npl) / 100
    provisions = (portfolio * predicted_npl * 0.5) / 100
    
    npl_multiplier = predicted_npl / current_npl
    rwa_increase = 1 + (npl_multiplier - 1) * 0.3
    new_rwa = portfolio * 0.8 * rwa_increase
    
    capital_depletion = expected_loss + provisions * 0.5
    current_capital = (portfolio * capital) / 100
    stressed_capital = current_capital - capital_depletion
    stressed_capital_ratio = (stressed_capital / new_rwa) * 100
    
    return {
        'predicted_npl': predicted_npl,
        'expected_loss': expected_loss,
        'provisions': provisions,
        'stressed_capital_ratio': stressed_capital_ratio,
        'capital_depletion': capital_depletion
    }

# Get predictions
rf_results = predict_impact(scenario_params, rf_model, scaler, portfolio_size, current_npl, capital_ratio)
gb_results = predict_impact(scenario_params, gb_model, scaler, portfolio_size, current_npl, capital_ratio)

results = {
    key: (rf_results[key] + gb_results[key]) / 2
    for key in rf_results.keys()
}

# Display metrics
st.markdown("---")
st.header(f"📈 Results for {scenario}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted NPL Ratio",
        f"{results['predicted_npl']:.2f}%",
        f"+{results['predicted_npl'] - current_npl:.2f}%"
    )

with col2:
    st.metric(
        "Stressed Capital Ratio",
        f"{results['stressed_capital_ratio']:.2f}%",
        f"{results['stressed_capital_ratio'] - capital_ratio:.2f}%"
    )

with col3:
    st.metric(
        "Expected Loss",
        f"${results['expected_loss']:.1f}M",
        f"{(results['expected_loss']/portfolio_size*100):.1f}% of portfolio"
    )

with col4:
    st.metric(
        "Capital Depletion",
        f"${results['capital_depletion']:.1f}M",
        f"-${results['capital_depletion']:.1f}M"
    )

# Regulatory compliance
st.markdown("---")
st.header("⚖️ Regulatory Compliance")

col1, col2, col3 = st.columns(3)

with col1:
    if results['stressed_capital_ratio'] >= 8:
        st.success("✅ PASS: Above Minimum (8%)")
    else:
        st.error("❌ FAIL: Below Minimum (8%)")

with col2:
    if results['stressed_capital_ratio'] >= 10.5:
        st.success("✅ PASS: Above Buffer (10.5%)")
    else:
        st.warning("⚠️ WARNING: Below Buffer (10.5%)")

with col3:
    if results['predicted_npl'] < 15:
        st.success("✅ NPL Below Threshold (15%)")
    else:
        st.error("❌ NPL Above Threshold (15%)")

# Visualizations
st.markdown("---")
st.header("📊 Visualizations")

# Run all scenarios for comparison
all_results = {}
for scen_name, scen_params in scenarios.items():
    rf_res = predict_impact(scen_params, rf_model, scaler, portfolio_size, current_npl, capital_ratio)
    gb_res = predict_impact(scen_params, gb_model, scaler, portfolio_size, current_npl, capital_ratio)
    all_results[scen_name] = {
        key: (rf_res[key] + gb_res[key]) / 2 for key in rf_res.keys()
    }

# Create comparison charts
col1, col2 = st.columns(2)

with col1:
    # NPL Comparison
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    scenario_names = list(all_results.keys())
    npl_values = [all_results[s]['predicted_npl'] for s in scenario_names]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    
    ax1.bar(scenario_names, npl_values, color=colors)
    ax1.axhline(y=current_npl, color='blue', linestyle='--', label='Current NPL', linewidth=2)
    ax1.axhline(y=15, color='red', linestyle='--', label='Threshold', linewidth=2)
    ax1.set_ylabel('NPL Ratio (%)', fontsize=11)
    ax1.set_title('NPL Ratios Across Scenarios', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    # Capital Comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    capital_values = [all_results[s]['stressed_capital_ratio'] for s in scenario_names]
    
    ax2.bar(scenario_names, capital_values, color=colors)
    ax2.axhline(y=8, color='red', linestyle='--', label='Minimum (8%)', linewidth=2)
    ax2.axhline(y=10.5, color='orange', linestyle='--', label='Buffer (10.5%)', linewidth=2)
    ax2.axhline(y=capital_ratio, color='blue', linestyle='--', label='Current', linewidth=2)
    ax2.set_ylabel('Capital Ratio (%)', fontsize=11)
    ax2.set_title('Capital Adequacy Across Scenarios', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig2)

# Feature Importance
st.markdown("---")
st.header("🎯 Feature Importance")

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
ax3.set_xlabel('Importance Score', fontsize=12)
ax3.set_title('Feature Importance for NPL Prediction', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# Scenario parameters
st.markdown("---")
st.header("📋 Scenario Parameters")

params_df = pd.DataFrame(scenarios).T
st.dataframe(params_df.style.format("{:.1f}"), use_container_width=True)

# Recommendations
st.markdown("---")
st.header("💡 Recommendations")

if results['stressed_capital_ratio'] < 8:
    st.error("🚨 **CRITICAL**: Capital ratio falls below regulatory minimum!")
    st.write(f"- Immediate capital raising required: ~${(8 * portfolio_size * 0.8 / 100) - (portfolio_size * results['stressed_capital_ratio'] / 100):.1f}M")
    st.write("- Consider capital conservation measures")
    st.write("- Review and potentially restrict dividend payments")
elif results['stressed_capital_ratio'] < 10.5:
    st.warning("⚠️ **WARNING**: Capital buffer inadequate")
    st.write("- Build capital buffer through retained earnings")
    st.write("- Review dividend policy")
    st.write("- Consider moderate capital raising")
else:
    st.success("✅ **HEALTHY**: Capital position adequate")
    st.write("- Continue monitoring economic indicators")
    st.write("- Maintain current risk management practices")

st.write(f"\n**Additional Recommendations:**")
st.write(f"- Recommended provision buffer: ${results['expected_loss'] * 1.2:.1f}M")
st.write(f"- Suggested capital target: {max(results['stressed_capital_ratio'] + 2, 12):.1f}%")
st.write(f"- Enhanced monitoring required if NPL exceeds {all_results['Moderate Stress']['predicted_npl']:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This model uses synthetic data for demonstration purposes. 
For production use, integrate with actual historical bank data.

Built with ❤️ using Python, scikit-learn, and Streamlit | 
[GitHub Repository](https://github.com/YOUR_USERNAME/bank-stress-testing-ml)
""")
