# 🏦 Bank Stress Testing Model - ML Implementation

A machine learning-based stress testing framework for small finance banks, predicting Non-Performing Loans (NPL), capital adequacy, and credit losses under various macroeconomic scenarios.

!

## 📊 Project Overview

This project demonstrates the application of machine learning techniques to financial risk management, specifically stress testing for banking portfolios. The model predicts how a bank's portfolio would perform under adverse economic conditions.

### Key Features

- **Multiple ML Models**: Random Forest and Gradient Boosting Regressors
- **Comprehensive Stress Scenarios**: Baseline, Moderate Stress, Severe Recession, Credit Crisis
- **Regulatory Compliance**: Basel III capital adequacy checks
- **Risk Quantification**: NPL prediction, capital depletion, expected losses
- **Interactive Visualizations**: 6+ professional charts and dashboards
- **Feature Importance Analysis**: Identifies key economic drivers of credit risk

## 🎯 Business Impact

- Quantifies potential losses under stress scenarios (up to $XX million)
- Ensures regulatory compliance (8% minimum capital ratio)
- Provides actionable recommendations for capital planning
- Identifies early warning indicators for risk management

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting)
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Cross-validation, RMSE, MAE, R²

## 📁 Project Structure

```
bank-stress-testing-ml/
│
├── bank_stress_test.py          # Main Python script
├── streamlit_app.py              # Interactive web app
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── data/                         # Data directory (if using real data)
│   └── sample_data.csv
│
├── outputs/                      # Generated outputs
│   ├── stress_test_dashboard.png
│   └── model_evaluation.png
│
└── docs/                         # Additional documentation
    └── methodology.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/bank-stress-testing-ml.git
cd bank-stress-testing-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python bank_stress_test.py
```

### Run as Web App

```bash
streamlit run streamlit_app.py
```

## 📈 Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | 0.45 | 0.32 | 0.94 |
| Gradient Boosting | 0.42 | 0.31 | 0.95 |

## 🧪 Stress Test Results

| Scenario | NPL Ratio | Capital Ratio | Expected Loss |
|----------|-----------|---------------|---------------|
| Baseline | 3.2% | 11.8% | $3.2M |
| Moderate Stress | 7.5% | 9.4% | $7.5M |
| Severe Recession | 12.1% | 7.2% | $12.1M |
| Credit Crisis | 15.2% | 6.1% | $15.2M |



## 📚 Methodology

### Data Generation
The model uses synthetic historical data simulating 500 quarters of macroeconomic conditions and loan performance. Features include:
- GDP Growth Rate
- Unemployment Rate
- Interest Rates
- Property Price Index
- Credit Spreads
- Liquidity Indicators

### Model Training
1. **Feature Engineering**: Standardization using StandardScaler
2. **Train-Test Split**: 80-20 split with random_state=42
3. **Model Selection**: Ensemble of Random Forest and Gradient Boosting
4. **Evaluation**: RMSE, MAE, R² metrics with cross-validation

### Stress Testing Framework
1. Define macroeconomic scenarios
2. Predict NPL ratios using trained models
3. Calculate capital impact and regulatory compliance
4. Generate recommendations

## 💡 Key Insights

- **Unemployment** is the strongest predictor of NPL (importance: 0.24)
- **GDP Growth** has significant negative correlation with NPL (-0.31)
- Capital adequacy falls below minimum in severe scenarios
- Recommended capital buffer: 12%+ to withstand stress

## 🔮 Future Enhancements

- [ ] Integration with real historical bank data
- [ ] Monte Carlo simulations for probabilistic forecasting
- [ ] LSTM/Neural Network models for time series prediction
- [ ] Portfolio segmentation analysis (retail, corporate, SME)
- [ ] Real-time dashboard with automated updates
- [ ] Integration with economic APIs for live data

## 📊 Use Cases

1. **Risk Management**: Quantify potential losses under adverse conditions
2. **Capital Planning**: Determine optimal capital levels
3. **Regulatory Reporting**: Basel III compliance documentation
4. **Board Presentations**: Executive-level risk reporting
5. **Strategic Planning**: Stress-informed decision making

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Aarti Navale**
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@Aarteaa](https://github.com/Aarteaa)
- Email: artinavale05@gmail.comm

## 🙏 Acknowledgments

- Basel Committee on Banking Supervision for regulatory frameworks
- scikit-learn community for excellent ML libraries
- Financial risk management literature and best practices

