from datetime import datetime
import MetaTrader5 as mt5

if not mt5.initialize():
    print(mt5.last_error())
    quit()
symbol = "XAUUSD.t"
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}'.looking for otther names>>>")
    all_symbols = mt5.symbols_get()
    if all_symbols:
        matches = [s.name for s in all_symbols if "XAU" in s.name.upper()]
        print(f"Available Gold symbols on your broker: {matches}")
        
        if matches:
            symbol = matches[0]
            print(f"Switching to '{symbol}'...")
            mt5.symbol_select(symbol, True)
        else:
            print("No Gold symbols found on your account.")
            mt5.shutdown()
            quit()

rates = mt5.copy_rates_range(
    symbol,
    mt5.TIMEFRAME_M5,
    datetime(2020, 1, 1),
    datetime(2026, 1, 1)
)
if rates is None:
    print(f"\nFailed to get rates for '{symbol}'.")
    print("Error details:", mt5.last_error())
    mt5.shutdown()
    quit()
print("Number of candles:", len(rates))

if len(rates) > 0:
    print("First candle:", datetime.fromtimestamp(rates[0]['time']))
    print("Last candle :", datetime.fromtimestamp(rates[-1]['time']))

mt5.shutdown()