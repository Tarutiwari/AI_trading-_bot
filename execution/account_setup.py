"""
Interactive Account Setup Wizard.
Asks user for capital, risk tolerance, and auto-configures the bot
for their specific account size.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG


class AccountSetupWizard:
    """
    Interactive setup that configures risk parameters based on user's account.
    Uses simple division-based lot sizing: Lot = Equity / Divisor
    """

    # Lot size divisor presets (lower = more aggressive, higher = safer)
    DIVISOR_PRESETS = {
        "1": {"name": "Ultra Safe",    "divisor": 1000, "desc": "Very small lots. $1000 account = 0.01 lot."},
        "2": {"name": "Conservative",  "divisor": 500,  "desc": "Safe for beginners. $500 account = 0.01 lot."},
        "3": {"name": "Moderate",      "divisor": 400,  "desc": "Balanced risk/reward. $400 = 0.01 lot."},
        "4": {"name": "Aggressive",    "divisor": 200,  "desc": "Higher lots. $200 = 0.01 lot. More risk."},
        "5": {"name": "Very Aggressive","divisor": 100,  "desc": "Maximum lots. $100 = 0.01 lot. High risk!"},
        "6": {"name": "Custom Divisor","divisor": None, "desc": "Enter your own divisor number."},
    }

    # Daily drawdown presets
    DD_PRESETS = {
        "1": {"name": "Tight",    "dd_pct": 3.0,  "desc": "Stop at 3% daily loss. Very safe."},
        "2": {"name": "Normal",   "dd_pct": 5.0,  "desc": "Stop at 5% daily loss. Balanced."},
        "3": {"name": "Wide",     "dd_pct": 10.0, "desc": "Stop at 10% daily loss. Aggressive."},
        "4": {"name": "Custom",   "dd_pct": None, "desc": "Enter your own daily drawdown %."},
    }

    def __init__(self, config=CONFIG):
        self.config = config
        self.user_config = {}

    def calculate_lot_from_divisor(self, equity: float, divisor: int) -> Tuple[float, float]:
        """
        Simple division-based lot sizing.
        Lot = Equity / Divisor
        Returns: (lot_size, min_lot_used)
        """
        raw_lot = equity / divisor
        # Round down to nearest 0.01
        lot_step = self.config.instrument.LOT_STEP
        rounded_lot = math.floor(raw_lot / lot_step) * lot_step
        
        # Enforce minimum lot
        min_lot = self.config.instrument.MIN_LOT
        final_lot = max(min_lot, rounded_lot)
        final_lot = round(final_lot, 2)
        
        was_clamped = final_lot == min_lot and raw_lot < min_lot
        return final_lot, was_clamped

    def calculate_sl_for_lot(self, equity: float, risk_pct: float, lot_size: float) -> float:
        """
        Calculate max SL distance (price) that keeps risk within budget.
        SL Distance = (Equity * Risk%) / (Lot Size * 100)
        """
        risk_amount = equity * (risk_pct / 100)
        if lot_size <= 0:
            return 0.0
        sl_distance = risk_amount / (lot_size * 100)  # 100 = contract size for XAUUSD
        return round(sl_distance, 2)

    def calculate_actual_risk(self, equity: float, lot_size: float, sl_distance: float) -> Dict:
        """
        Calculate actual risk in dollars and percentage.
        """
        actual_risk = lot_size * sl_distance * 100  # lot * price_distance * contract_size
        risk_pct = (actual_risk / equity * 100) if equity > 0 else 0
        return {
            "risk_dollars": round(actual_risk, 2),
            "risk_pct": round(risk_pct, 2),
            "is_over_risk": risk_pct > 2.0,  # Warn if > 2% actual risk
        }

    def run_interactive_setup(self) -> Dict:
        """Run the interactive setup wizard. Returns config dict."""
        print("\n" + "=" * 60)
        print("  🤖 MAMBA GOLD TRADING BOT - ACCOUNT SETUP")
        print("=" * 60)
        print("  Formula: Lot Size = Your Capital / Divisor")
        print("  Example: $500 / 500 = 0.01 lot")
        print("           $1000 / 500 = 0.02 lot")

        # Step 1: Get account equity
        while True:
            try:
                equity_input = input("\n  💰 Enter your account equity in USD (e.g. 500): $").strip()
                equity = float(equity_input.replace(",", "").replace("$", ""))
                if equity <= 0:
                    print("  ❌ Equity must be positive!")
                    continue
                if equity < 50:
                    print(f"  ⚠️  ${equity:.0f} is very small. Minimum recommended is $100.")
                    confirm = input("  Continue anyway? (y/n): ").strip().lower()
                    if confirm != "y":
                        continue
                break
            except ValueError:
                print("  ❌ Please enter a valid number!")

        # Step 2: Show lot preview for different divisors
        print(f"\n  📊 Lot Size Preview for ${equity:,.0f}:")
        print(f"  {'─' * 55}")
        print(f"  {'Divisor':>10s} {'Lot Size':>10s} {'Risk/Trade':>12s} {'Example':>20s}")
        print(f"  {'─' * 55}")
        
        for div_key, div_preset in self.DIVISOR_PRESETS.items():
            if div_preset["divisor"] is None:
                continue
            lot, _ = self.calculate_lot_from_divisor(equity, div_preset["divisor"])
            risk = lot * 1.5 * 100  # Approximate risk at 1.5 ATR SL
            print(f"  {div_preset['name']:>10s}: /{div_preset['divisor']:<6d} = {lot:.2f} lot  ~${risk:.2f} risk")
        print(f"  {'─' * 55}")

        # Step 3: Choose divisor
        print(f"\n  🎯 Choose Lot Size aggressiveness:")
        for key, preset in self.DIVISOR_PRESETS.items():
            print(f"  [{key}] {preset['name']:18s} - {preset['desc']}")

        while True:
            choice = input(f"\n  Enter choice (1-6): ").strip()
            if choice in self.DIVISOR_PRESETS:
                break
            print("  ❌ Invalid choice!")

        divisor_preset = self.DIVISOR_PRESETS[choice]

        # Handle custom divisor
        if divisor_preset["divisor"] is None:
            while True:
                try:
                    divisor = int(input("  Enter divisor (e.g. 300, 500, 1000): ").strip())
                    if 50 <= divisor <= 5000:
                        break
                    print("  ❌ Divisor must be between 50 and 5000!")
                except ValueError:
                    print("  ❌ Enter a valid number!")
        else:
            divisor = divisor_preset["divisor"]

        # Calculate lot size
        lot_size, was_clamped = self.calculate_lot_from_divisor(equity, divisor)

        # Step 4: Choose daily drawdown limit
        print(f"\n  🛡️ Choose Daily Drawdown Limit:")
        for key, preset in self.DD_PRESETS.items():
            if preset["dd_pct"] is not None:
                dd_amount = equity * (preset["dd_pct"] / 100)
                print(f"  [{key}] {preset['name']:12s} - {preset['dd_pct']:.0f}% = ${dd_amount:.2f} max daily loss")
            else:
                print(f"  [{key}] {preset['name']:12s} - {preset['desc']}")

        while True:
            dd_choice = input(f"\n  Enter choice (1-4): ").strip()
            if dd_choice in self.DD_PRESETS:
                break
            print("  ❌ Invalid choice!")

        dd_preset = self.DD_PRESETS[dd_choice]
        if dd_preset["dd_pct"] is None:
            daily_dd_pct = self._get_custom_value("daily drawdown %", 1.0, 20.0, 5.0)
        else:
            daily_dd_pct = dd_preset["dd_pct"]

        # Step 5: Calculate all parameters
        daily_dd_amount = equity * (daily_dd_pct / 100)
        risk_per_lot = lot_size * 1.5 * 100  # Approximate at 1.5 ATR SL
        risk_pct = (risk_per_lot / equity * 100) if equity > 0 else 0

        print(f"\n  {'═' * 55}")
        print(f"  📋 YOUR TRADING CONFIGURATION")
        print(f"  {'═' * 55}")
        print(f"  Account Equity      : ${equity:,.2f}")
        print(f"  Formula             : ${equity:,.0f} / {divisor} = {lot_size} lot")
        print(f"  Lot Size            : {lot_size}")
        print(f"  Risk per Trade      : ~${risk_per_lot:.2f} ({risk_pct:.1f}% of account)")
        print(f"  Daily Loss Limit    : {daily_dd_pct:.1f}% = ${daily_dd_amount:.2f}")
        
        if was_clamped:
            print(f"  ⚠️  Lot clamped to minimum {self.config.instrument.MIN_LOT} (account too small for this divisor)")
        
        print(f"  {'═' * 55}")

        # Step 6: Confirm
        confirm = input("  ✅ Apply this configuration? (y/n): ").strip().lower()
        if confirm != "y":
            print("  ❌ Setup cancelled. Run again to restart.")
            return {}

        # Step 7: Build and save config
        self.user_config = {
            "equity": equity,
            "divisor": divisor,
            "lot_size": lot_size,
            "risk_per_trade": risk_per_lot,
            "risk_pct": risk_pct,
            "daily_dd_pct": daily_dd_pct,
            "daily_dd_amount": daily_dd_amount,
        }

        # Apply to live config
        self._apply_to_config()

        # Save to file
        save_path = self.config.paths.CHECKPOINTS_DIR / "account_config.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.user_config, f, indent=2)
        print(f"\n  💾 Configuration saved -> {save_path.name}")

        print(f"\n  🚀 Bot is ready!")
        print(f"     Capital: ${equity:,.0f} | Lot: {lot_size} | Daily limit: {daily_dd_pct}% (${daily_dd_amount:.2f})")

        return self.user_config

    def _get_custom_value(self, name: str, min_val: float, max_val: float, default: float) -> float:
        """Get a custom percentage value from user."""
        while True:
            try:
                val = input(f"  Enter {name} [{min_val}-{max_val}%] (default {default}%): ").strip()
                if val == "":
                    return default
                val = float(val.replace("%", ""))
                if min_val <= val <= max_val:
                    return val
                print(f"  ❌ Must be between {min_val}% and {max_val}%!")
            except ValueError:
                print("  ❌ Please enter a valid number!")

    def _apply_to_config(self):
        """Apply user settings to the live config object."""
        if not self.user_config:
            return

        # Update risk management config
        self.config.risk.MAX_DAILY_DRAWDOWN_PCT = self.user_config["daily_dd_pct"] / 100
        
        # Set lot size as both min and max for consistency
        lot = self.user_config["lot_size"]
        self.config.instrument.MIN_LOT = lot
        self.config.instrument.MAX_LOT = lot

        # Store divisor for reference
        self.config._lot_divisor = self.user_config.get("divisor", 400)

        print(f"  [CONFIG] Applied: lot={lot}, dd={self.config.risk.MAX_DAILY_DRAWDOWN_PCT*100:.1f}%")

    def load_saved_setup(self) -> Dict:
        """Load previously saved account config if it exists."""
        save_path = self.config.paths.CHECKPOINTS_DIR / "account_config.json"
        if save_path.exists():
            with open(save_path, "r") as f:
                self.user_config = json.load(f)
            self._apply_to_config()
            
            # Handle both old and new config formats
            if "divisor" in self.user_config:
                print(f"[SETUP] Loaded: ${self.user_config['equity']:,.0f} / {self.user_config['divisor']} = {self.user_config['lot_size']} lot")
            else:
                print(f"[SETUP] Loaded: ${self.user_config['equity']:,.0f} account")
            return self.user_config
        return {}


def quick_setup(equity: float, divisor: int = 400, daily_dd_pct: float = 5.0) -> Dict:
    """
    Non-interactive quick setup for programmatic use.
    Example: quick_setup(500, divisor=500, daily_dd_pct=3.0)
    
    Formula: Lot = Equity / Divisor
    $500 / 500 = 0.01 lot
    $1000 / 500 = 0.02 lot
    """
    wizard = AccountSetupWizard()

    lot_size, was_clamped = wizard.calculate_lot_from_divisor(equity, divisor)
    daily_dd_amount = equity * (daily_dd_pct / 100)
    risk_per_lot = lot_size * 1.5 * 100  # Approximate at 1.5 ATR SL

    config = {
        "equity": equity,
        "divisor": divisor,
        "lot_size": lot_size,
        "risk_per_trade": risk_per_lot,
        "risk_pct": (risk_per_lot / equity * 100) if equity > 0 else 0,
        "daily_dd_pct": daily_dd_pct,
        "daily_dd_amount": daily_dd_amount,
    }

    wizard.user_config = config
    wizard._apply_to_config()

    print(f"[SETUP] Quick config: ${equity:,.0f} / {divisor} = {lot_size} lot | "
          f"Daily: {daily_dd_pct}% (${daily_dd_amount:.2f})")

    return config


if __name__ == "__main__":
    wizard = AccountSetupWizard()
    result = wizard.run_interactive_setup()
    if result:
        print("\n  Setup complete! Run 'python main_live.py' to start trading.")
