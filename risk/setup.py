import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from fpdf import FPDF

class RiskEngine:
    def __init__(self, tickers, eur_stocks, threshold_95=0.04, threshold_99=0.06):
        self.tickers = tickers
        self.eur_stocks = eur_stocks
        self.fx_pair = "EURUSD=X"
        self.fx_commission = 0.0002 
        self.t95 = threshold_95
        self.t99 = threshold_99
        self.adj_returns = None

    def fetch_data(self, start_date="2024-01-01"):
        """Downloads data and adjusts for FX + Commission."""
        print(f"--- Fetching Data for {self.tickers} ---")
        data = yf.download(self.tickers + [self.fx_pair], start=start_date)['Close']
        
        prices = data[self.tickers].fillna(method='ffill')
        fx = data[self.fx_pair].fillna(method='ffill')
        
        for stock in self.eur_stocks:
            if stock in prices.columns:
                prices[stock] = prices[stock] * (fx * (1 + self.fx_commission))
        
        self.adj_returns = prices.pct_change().dropna()
        print("--- Data Processing Complete ---")

    def get_metrics_df(self):
        """Calculates tiered VaR and Tail Stress metrics."""
        results = []
        for ticker in self.tickers:
            rets = self.adj_returns[ticker]
            
            h95 = np.percentile(rets, 5)
            h99 = np.percentile(rets, 1)
            
            tail_stress = h99 / h95 if h95 != 0 else 0
            
            alert_95 = "WARNING" if abs(h95) > self.t95 else "OK"
            alert_99 = "CRITICAL" if abs(h99) > self.t99 else "OK"
            
            results.append({
                "Ticker": ticker,
                "VaR 95%": f"{h95:.2%}",
                "Alert 95": alert_95,
                "VaR 99%": f"{h99:.2%}",
                "Alert 99": alert_99,
                "Tail Stress": f"{tail_stress:.2f}x"
            })
        return pd.DataFrame(results)

    def export_to_pdf(self, df, filename="Risk_Report.pdf"):
        """Generates a professional PDF report."""
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(190, 10, "Portfolio Risk Setup Report", ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100) 
        pdf.multi_cell(0, 5, f"Confidence Levels: 95% and 99% Historical VaR Analysis\n"
                             f"Thresholds: 95% Limit @ {self.t95:.1%}, 99% Limit @ {self.t99:.1%}\n"
                             f"FX Adjustment: 0.02% commission included in price returns.\n"
                             f"Base Currency: USD\n")
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 9)
        pdf.set_text_color(0, 0, 0)
        cols = df.columns
        column_width = 190 / len(cols)
        
        for col in cols:
            pdf.cell(column_width, 10, col, border=1, align='C')
        pdf.ln()

        pdf.set_font("Arial", size=9)
        for _, row in df.iterrows():
            for col in cols:
                val = str(row[col])
                pdf.cell(column_width, 10, val, border=1, align='C')
            pdf.ln()

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 8)
        pdf.multi_cell(0, 5, "Note: 'Tail Stress' indicates the multiplier of risk when moving from a "
                             "95% to a 99% confidence interval. Ratios above 1.5x suggest high leptokurtosis (fat tails).")

        pdf.output(filename)
        print(f"--- PDF Report Generated: {filename} ---")


if __name__ == "__main__":
    assets = ["MU", "MSFT", "LLY", "MC", "ASML"]
    euro_assets = ["MC", "ASML"]

    engine = RiskEngine(assets, euro_assets, threshold_95=0.04, threshold_99=0.06)

    engine.fetch_data()
    metrics_df = engine.get_metrics_df()
    
    print("\n--- Terminal Preview ---")
    print(metrics_df.to_string(index=False))
    engine.export_to_pdf(metrics_df)