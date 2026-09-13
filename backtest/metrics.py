"""
Quantitative Metrics & Institutional Performance Tearsheet Generator.
Calculates Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, and Expectancy.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Computes institutional-grade quantitative trading performance statistics.
    """

    @staticmethod
    def calculate_all_metrics(
        equity_series: pd.Series,
        trade_records: List[Dict[str, any]],
        initial_capital: float = 10_000.0,
    ) -> Dict[str, any]:
        """
        Computes summary tearsheet metrics from equity curve and trade logs.
        """
        if equity_series.empty or len(equity_series) < 2:
            return {"error": "Insufficient equity data."}

        # 1. Equity-Based Metrics
        returns = equity_series.pct_change().dropna()
        final_equity = float(equity_series.iloc[-1])
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

        # Maximum Drawdown (Peak to Trough)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = float(abs(drawdowns.min())) * 100.0

        # Annualized Sharpe & Sortino (Assuming 252 trading days, ~72 5-min bars/day = 18,144 bars/yr)
        bars_per_year = 252 * 72
        mean_ret = returns.mean()
        std_ret = returns.std()
        downside_std = returns[returns < 0].std()

        sharpe = float((mean_ret / (std_ret + 1e-8)) * np.sqrt(bars_per_year)) if std_ret > 0 else 0.0
        sortino = float((mean_ret / (downside_std + 1e-8)) * np.sqrt(bars_per_year)) if downside_std > 0 else 0.0
        calmar = float((total_return_pct / (max_drawdown_pct + 1e-8))) if max_drawdown_pct > 0 else 0.0

        # 2. Trade-Based Metrics
        total_trades = len(trade_records)
        if total_trades > 0:
            pnl_list = [t["pnl"] for t in trade_records]
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]

            win_rate = (len(wins) / total_trades) * 100.0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
            expectancy = np.mean(pnl_list)
        else:
            win_rate = gross_profit = gross_loss = avg_win = avg_loss = win_loss_ratio = expectancy = 0.0
            profit_factor = 0.0

        metrics = {
            "Initial Capital ($)": f"${initial_capital:,.2f}",
            "Final Equity ($)": f"${final_equity:,.2f}",
            "Total Return (%)": f"{total_return_pct:+.2f}%",
            "Max Drawdown (%)": f"{max_drawdown_pct:.2f}%",
            "Annualized Sharpe": f"{sharpe:.2f}",
            "Annualized Sortino": f"{sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Total Trades": total_trades,
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "Avg Win ($)": f"${avg_win:,.2f}",
            "Avg Loss ($)": f"${avg_loss:,.2f}",
            "Win/Loss Ratio": f"{win_loss_ratio:.2f}",
            "Expectancy ($/trade)": f"${expectancy:,.2f}",
        }
        return metrics

    @staticmethod
    def print_tearsheet(metrics: Dict[str, any]) -> None:
        """Formats and prints an institutional performance report."""
        print("\n=======================================================")
        print("          INSTITUTIONAL PERFORMANCE TEARSHEET         ")
        print("=======================================================")
        for key, value in metrics.items():
            print(f"  {key:<24}: {value}")
        print("=======================================================\n")
