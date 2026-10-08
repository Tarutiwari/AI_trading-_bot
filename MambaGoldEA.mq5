//+------------------------------------------------------------------+
//|                                                   MambaGoldEA.mq5 |
//|  Native Expert Advisor that executes signals produced by the      |
//|  Mamba gold model via `python deploy_bot.py --signal-export`.     |
//|                                                                  |
//|  SETUP (3 steps):                                                |
//|   1. In MetaEditor: File > New > Expert Advisor > paste this code|
//|      and press F5 (compile). Or copy this file into your          |
//|      MQL5\Experts\ folder and compile.                            |
//|   2. Attach the EA to the XAUUSD M5 chart of the SAME broker     |
//|      symbol the Python bot trades.                               |
//|   3. On the PC, run:  python deploy_bot.py --signal-export        |
//|      (Python never trades in this mode; the EA executes).        |
//|                                                                  |
//|  The EA reads logs/..\mamba_signal.csv from the MT5 Common files |
//|  folder each poll cycle and then, on its own:                    |
//|    * opens BUY/SELL with server-side SL/TP (1.5x ATR each)       |
//|    * sizes lots at RiskPercent% of equity                        |
//|    * breakeven at 1R + trailing stop every tick                   |
//|    * force-closes after MaxHoldMin (default 120 min = 24 M5 bars)|
//|    * reverses when the model signal flips                         |
//|    * halts + closes all at DailyDDPct / consecutive losses        |
//|    * blocks entries on news freeze, wide spread, stale signal     |
//|                                                                  |
//|  SAFETY: NEVER run the EA and `python deploy_bot.py` (live mode) |
//|  at the same time — that would double every order.               |
//|  Start on a DEMO account. The model underperforms out-of-sample. |
//+------------------------------------------------------------------+
#property version   "1.00"
#property description "Mamba XAUUSD signal executor — bridge to deploy_bot.py --signal-export"

#include <Trade\Trade.mqh>

input group "=== Signal bridge ==="
input string InpSignalFile       = "mamba_signal.csv"; // Signal file name
input bool   InpUseCommonDir     = true;               // Read Common files folder (recommended)
input int    InpMaxAgeSec        = 120;                // Discard signals older than N seconds
input int    InpPollSeconds      = 15;                 // Seconds between signal reads / entries
input bool   InpRequireSymMatch  = true;               // Signal symbol must match this chart

input group "=== Risk ==="
input double InpRiskPercent      = 1.0;                // Equity risk per trade (%)
input double InpDailyDDPct       = 3.0;                // Daily drawdown % -> halt for the day
input int    InpMaxConsecLoss    = 5;                  // Consecutive losses -> halt (0 = off)
input int    InpMaxOpen          = 1;                  // Max simultaneous positions
input double InpMinLot           = 0.0;                // Min lot (0 = broker min)
input double InpMaxLot           = 5.0;                // Max lot (also capped by broker)
input double InpMaxSpreadPts     = 40.0;               // Max spread in points for entry
input double InpMinConfidence    = 0.50;               // Min signal confidence to trade

input group "=== Exits ==="
input double InpSLATR            = 1.5;                // Stop-loss  = x * ATR(14,M5)
input double InpTPATR            = 1.5;                // Take-profit = x * ATR(14,M5)
input int    InpMaxHoldMin       = 120;                // Force-close after N minutes (0 = off)
input bool   InpTrailOn          = true;               // Enable trailing stop
input double InpTrailATR         = 1.5;                // Trail distance = x * ATR
input bool   InpBreakeven        = true;               // Move SL to breakeven at 1R
input double InpBE_R             = 1.0;                // Breakeven at x R profit
input bool   InpReverse          = true;               // Close + flip when signal reverses
input int    InpSlippagePts      = 20;                 // Max slippage in points

input group "=== Identification ==="
input ulong  InpMagic            = 777001;             // Magic number (match Python bot)
input string InpComment          = "Mamba_XAUUSD_Bot"; // Order comment
input bool   InpVerbose          = false;              // Log every poll cycle

//--- signal container
struct SignalData
{
   bool   valid;
   long   epoch;
   string action;
   string symbol;
   double conf;
   double pBuy;
   double pSell;
   double pNeutral;
   int    regime;
   double atr;
   bool   newsFrozen;
};

CTrade         g_trade;
int            g_atrHandle = INVALID_HANDLE;
double         g_atr[];
datetime       g_dayStart      = 0;
double         g_dayStartEquity = 0;
bool           g_halted        = false;
ulong          g_nextPollMs    = 0;
ulong          g_lastWarnMs    = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   if(StringLen(InpSignalFile) == 0)
   {
      Print("[MAMBA EA] ERROR: empty signal file name");
      return(INIT_PARAMETERS_INCORRECT);
   }

   g_atrHandle = iATR(_Symbol, PERIOD_M5, 14);
   if(g_atrHandle == INVALID_HANDLE)
   {
      Print("[MAMBA EA] ERROR: iATR failed, error=", GetLastError());
      return(INIT_FAILED);
   }
   ArraySetAsSeries(g_atr, true);

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePts);

   PrintFormat("[MAMBA EA] initialized on %s | magic=%d | file=%s(%s) | risk=%.1f%% | conf>=%.2f",
               _Symbol, InpMagic, InpSignalFile,
               InpUseCommonDir ? "Common" : "MQL5\\Files",
               InpRiskPercent, InpMinConfidence);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE)
      IndicatorRelease(g_atrHandle);
}

//+------------------------------------------------------------------+
//| Read the key=value bridge file written by deploy_bot.py          |
//+------------------------------------------------------------------+
bool ReadSignal(SignalData &s)
{
   s.valid       = false;
   s.epoch       = 0;
   s.action      = "";
   s.symbol      = "";
   s.conf        = 0;
   s.pBuy        = 0;
   s.pSell       = 0;
   s.pNeutral    = 0;
   s.regime      = -1;
   s.atr         = 0;
   s.newsFrozen  = false;

   int flags = FILE_TXT | FILE_READ | FILE_ANSI | FILE_SHARE_READ;
   if(InpUseCommonDir)
      flags |= FILE_COMMON;

   int h = FileOpen(InpSignalFile, flags);
   if(h == INVALID_HANDLE)
      return(false);

   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      if(StringLen(line) == 0)
         continue;
      int p = StringFind(line, "=");
      if(p < 0)
         continue;
      string k = StringSubstr(line, 0, p);
      string v = StringSubstr(line, p + 1);
      StringTrimLeft(k);
      StringTrimRight(k);
      StringTrimLeft(v);
      StringTrimRight(v);

      if(k == "action")         s.action      = v;
      else if(k == "epoch")     s.epoch       = StringToInteger(v);
      else if(k == "symbol")    s.symbol      = v;
      else if(k == "confidence")s.conf        = StringToDouble(v);
      else if(k == "p_buy")     s.pBuy        = StringToDouble(v);
      else if(k == "p_sell")    s.pSell       = StringToDouble(v);
      else if(k == "p_neutral") s.pNeutral    = StringToDouble(v);
      else if(k == "regime")    s.regime      = (int)StringToInteger(v);
      else if(k == "atr")       s.atr         = StringToDouble(v);
      else if(k == "news_frozen") s.newsFrozen = (StringToInteger(v) != 0);
   }
   FileClose(h);

   s.valid = (s.epoch > 0 && StringLen(s.action) > 0);
   return(s.valid);
}

//+------------------------------------------------------------------+
//| Throttled warning (at most once per minute)                       |
//+------------------------------------------------------------------+
void WarnThrottled(const string msg)
{
   ulong now = GetTickCount64();
   if(now - g_lastWarnMs < 60000)
      return;
   g_lastWarnMs = now;
   Print("[MAMBA EA] ", msg);
}

//+------------------------------------------------------------------+
//| Count positions opened by this EA on this symbol                  |
//+------------------------------------------------------------------+
int CountMine()
{
   int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      n++;
   }
   return(n);
}

//+------------------------------------------------------------------+
void CloseAllMine(const string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      if(g_trade.PositionClose(tk, InpSlippagePts))
         PrintFormat("[MAMBA EA] CLOSED #%I64u (%s)", tk, reason);
      else
         PrintFormat("[MAMBA EA] close #%I64u FAILED retcode=%d %s",
                     tk, g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| New-day equity baseline + consecutive-loss recount                |
//+------------------------------------------------------------------+
void ResetIfNewDay()
{
   datetime dayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   if(dayStart == g_dayStart)
      return;

   g_dayStart       = dayStart;
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_halted         = false;
   int consec       = CountConsecLosses();
   if(InpMaxConsecLoss > 0 && consec >= InpMaxConsecLoss)
   {
      g_halted = true;
      PrintFormat("[MAMBA EA] new day starts with %d consecutive losses — halted", consec);
   }
   if(consec > 0)
      PrintFormat("[MAMBA EA] new day: equity $%.2f, carry-over consecutive losses=%d",
                  g_dayStartEquity, consec);
}

//+------------------------------------------------------------------+
//| Consecutive losing closed trades today (this magic, this symbol)  |
//+------------------------------------------------------------------+
int CountConsecLosses()
{
   if(g_dayStart == 0 || !HistorySelect(g_dayStart, TimeCurrent()))
      return(0);

   int consec = 0;
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
   {
      ulong tk = HistoryDealGetTicket(i);
      if(tk == 0)
         continue;
      if(HistoryDealGetInteger(tk, DEAL_MAGIC) != (long)InpMagic)
         continue;
      if(HistoryDealGetString(tk, DEAL_SYMBOL) != _Symbol)
         continue;
      long entry = HistoryDealGetInteger(tk, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
         continue;
      double pnl = HistoryDealGetDouble(tk, DEAL_PROFIT)
                 + HistoryDealGetDouble(tk, DEAL_COMMISSION)
                 + HistoryDealGetDouble(tk, DEAL_SWAP);
      if(pnl < 0)
         consec++;
      else
         break;
   }
   return(consec);
}

//+------------------------------------------------------------------+
//| Daily drawdown circuit breaker                                    |
//+------------------------------------------------------------------+
void UpdateHalt()
{
   if(g_dayStartEquity <= 0)
      return;
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
   if(dd >= InpDailyDDPct)
   {
      if(!g_halted)
      {
         PrintFormat("[MAMBA EA] DAILY DRAWDOWN %.2f%% >= %.2f%% — HALT + close all",
                     dd, InpDailyDDPct);
         if(InpMaxConsecLoss > 0)
         {
            int consec = CountConsecLosses();
            if(consec >= InpMaxConsecLoss)
               PrintFormat("[MAMBA EA] also at %d consecutive losses", consec);
         }
      }
      g_halted = true;
      if(CountMine() > 0)
         CloseAllMine("daily drawdown halt");
   }
}

//+------------------------------------------------------------------+
//| Per-tick position management: max-hold close, breakeven, trailing |
//+------------------------------------------------------------------+
void ManageMine(const double bid, const double ask, const double atr)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;

      bool   isBuy  = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double open   = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl     = PositionGetDouble(POSITION_SL);
      double tp     = PositionGetDouble(POSITION_TP);
      datetime openT = (datetime)PositionGetInteger(POSITION_TIME);

      // --- max-holding force close -----------------------------------
      if(InpMaxHoldMin > 0 && TimeCurrent() - openT >= (long)InpMaxHoldMin * 60)
      {
         if(g_trade.PositionClose(tk, InpSlippagePts))
            PrintFormat("[MAMBA EA] CLOSED #%I64u max-hold %d min reached", tk, InpMaxHoldMin);
         continue;
      }

      double cur = isBuy ? bid : ask;
      double initialRisk = (sl > 0) ? MathAbs(open - sl) : atr * InpSLATR;
      if(initialRisk <= 0)
         initialRisk = atr * InpSLATR;

      // --- breakeven at 1R -------------------------------------------
      if(isBuy)
      {
         if(InpBreakeven && cur - open >= InpBE_R * initialRisk && sl < open - 1e-12)
         {
            double ns = NormalizeDouble(open + 10 * _Point, _Digits);
            if(g_trade.PositionModify(tk, ns, tp))
               PrintFormat("[MAMBA EA] #%I64u breakeven SL -> %.5f", tk, ns);
            sl = ns;
         }
         if(InpTrailOn && sl >= open - 1e-12)
         {
            double ns = NormalizeDouble(cur - InpTrailATR * atr, _Digits);
            if(ns > sl + 1e-12 && g_trade.PositionModify(tk, ns, tp))
               PrintFormat("[MAMBA EA] #%I64u trail SL -> %.5f", tk, ns);
         }
      }
      else
      {
         if(InpBreakeven && open - cur >= InpBE_R * initialRisk && (sl > open + 1e-12 || sl <= 0))
         {
            double ns = NormalizeDouble(open - 10 * _Point, _Digits);
            if(g_trade.PositionModify(tk, ns, tp))
               PrintFormat("[MAMBA EA] #%I64u breakeven SL -> %.5f", tk, ns);
            sl = ns;
         }
         if(InpTrailOn && sl > 0 && sl <= open + 1e-12)
         {
            double ns = NormalizeDouble(cur + InpTrailATR * atr, _Digits);
            if(ns < sl - 1e-12 && g_trade.PositionModify(tk, ns, tp))
               PrintFormat("[MAMBA EA] #%I64u trail SL -> %.5f", tk, ns);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close our position when the model signal is on the other side     |
//+------------------------------------------------------------------+
void ReverseOnSignal(const SignalData &sig)
{
   if(!InpReverse)
      return;
   if(sig.conf < InpMinConfidence)
      return;
   bool wantBuy  = (StringCompare(sig.action, "BUY", false) == 0);
   bool wantSell = (StringCompare(sig.action, "SELL", false) == 0);
   if(!wantBuy && !wantSell)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      bool isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      if((isBuy && wantSell) || (!isBuy && wantBuy))
      {
         if(g_trade.PositionClose(tk, InpSlippagePts))
            PrintFormat("[MAMBA EA] CLOSED #%I64u model reversed -> %s (%.1f%%)",
                        tk, sig.action, sig.conf * 100.0);
      }
   }
}

//+------------------------------------------------------------------+
//| Filling mode supported by this broker/symbol                      |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING ChooseFilling()
{
   long mode = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((mode & 1) != 0)   // SYMBOL_FILLING_FOK
      return(ORDER_FILLING_FOK);
   if((mode & 2) != 0)   // SYMBOL_FILLING_IOC
      return(ORDER_FILLING_IOC);
   return(ORDER_FILLING_RETURN);
}

//+------------------------------------------------------------------+
//| Risk-based lot size (mirrors execution/risk_manager.py)           |
//+------------------------------------------------------------------+
double CalcLots(const double slDist)
{
   double eq        = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney = eq * InpRiskPercent / 100.0;
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double vmin      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax      = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vstep     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vlimit    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_LIMIT);
   if(vstep <= 0)
      vstep = 0.01;

   double raw = 0;
   if(tickSize > 0 && tickValue > 0 && slDist > 0)
      raw = riskMoney / (slDist / tickSize * tickValue);

   double lots = raw;
   double minLot = (InpMinLot > 0 ? InpMinLot : vmin);
   if(vmin > 0 && minLot < vmin)
      minLot = vmin;
   if(lots < minLot)
   {
      if(raw > 0 && raw < minLot)
         WarnThrottled("broker minimum lot forces over-risking — consider a larger account");
      lots = minLot;
   }

   lots = MathFloor(lots / vstep) * vstep;

   double cap = InpMaxLot;
   if(vmax > 0 && vmax < cap)
      cap = vmax;
   if(vlimit > 0 && vlimit < cap)
      cap = vlimit;
   if(lots > cap)
   {
      WarnThrottled("calculated lot exceeds cap — capped (over-risking)");
      lots = cap;
   }
   return(NormalizeDouble(lots, 2));
}

//+------------------------------------------------------------------+
//| Open a new position from the signal                               |
//+------------------------------------------------------------------+
void TryEntry(const SignalData &sig, const double atr, const double spreadPts)
{
   if(g_halted)
      return;
   if(CountMine() >= InpMaxOpen)
      return;

   bool wantBuy  = (StringCompare(sig.action, "BUY", false) == 0);
   bool wantSell = (StringCompare(sig.action, "SELL", false) == 0);
   if(!wantBuy && !wantSell)
      return;
   if(sig.conf < InpMinConfidence)
   {
      if(InpVerbose)
         PrintFormat("[MAMBA EA] skip %s: confidence %.2f < %.2f",
                     sig.action, sig.conf, InpMinConfidence);
      return;
   }
   if(sig.newsFrozen)
   {
      WarnThrottled("news freeze active — entry blocked");
      return;
   }
   if(spreadPts > InpMaxSpreadPts)
   {
      if(InpVerbose)
         PrintFormat("[MAMBA EA] skip: spread %.1f pts > %.0f", spreadPts, InpMaxSpreadPts);
      return;
   }
   if(atr <= 0)
      return;

   double slDist = InpSLATR * atr;
   double tpDist = InpTPATR * atr;
   long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = stopsLevel * _Point;
   if(minDist > 0)
   {
      if(slDist < minDist) slDist = minDist;
      if(tpDist < minDist) tpDist = minDist;
   }

   double lots = CalcLots(slDist);
   double bid  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask  = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double price, sl, tp;
   bool ok;

   g_trade.SetTypeFilling(ChooseFilling());

   if(wantBuy)
   {
      price = ask;
      sl    = NormalizeDouble(price - slDist, _Digits);
      tp    = NormalizeDouble(price + tpDist, _Digits);
      ok    = g_trade.Buy(lots, _Symbol, price, sl, tp, InpComment);
   }
   else
   {
      price = bid;
      sl    = NormalizeDouble(price + slDist, _Digits);
      tp    = NormalizeDouble(price - tpDist, _Digits);
      ok    = g_trade.Sell(lots, _Symbol, price, sl, tp, InpComment);
   }

   if(ok)
      PrintFormat("[MAMBA EA] OPEN %s %.2f lots %s @ %.5f | SL %.5f TP %.5f | conf %.1f%% | ATR %.2f | spread %.1f pts",
                  sig.action, lots, _Symbol, price, sl, tp, sig.conf * 100.0, atr, spreadPts);
   else
      PrintFormat("[MAMBA EA] ORDER FAILED retcode=%d %s",
                  g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
void OnTick()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(bid <= 0 || ask <= 0)
      return;
   double spreadPts = (ask - bid) / _Point;

   if(CopyBuffer(g_atrHandle, 0, 0, 1, g_atr) < 1)
   {
      WarnThrottled("ATR buffer unavailable");
      return;
   }
   double atr = g_atr[0];

   ResetIfNewDay();
   UpdateHalt();
   ManageMine(bid, ask, atr);

   // --- signal-driven decisions at poll cadence ----------------------
   if(GetTickCount64() < g_nextPollMs)
      return;
   g_nextPollMs = GetTickCount64() + (ulong)InpPollSeconds * 1000;

   SignalData sig;
   if(!ReadSignal(sig))
   {
      WarnThrottled("signal file missing/unreadable — is `python deploy_bot.py --signal-export` running?");
      return;
   }

   long age = (long)TimeGMT() - (long)sig.epoch;
   if(age > InpMaxAgeSec || age < -300)
   {
      WarnThrottled("stale signal (age " + IntegerToString(age) + "s) — not trading");
      return;
   }
   if(InpRequireSymMatch && StringLen(sig.symbol) > 0 && sig.symbol != _Symbol)
   {
      WarnThrottled("signal symbol '" + sig.symbol + "' != chart '" + _Symbol + "' — not trading");
      return;
   }

   if(InpVerbose)
      PrintFormat("[MAMBA EA] %s conf %.1f%% p(B/S/N) %.2f/%.2f/%.2f regime %d age %lds open %d%s",
                  sig.action, sig.conf * 100.0, sig.pBuy, sig.pSell, sig.pNeutral,
                  sig.regime, age, CountMine(),
                  sig.newsFrozen ? " [NEWS FREEZE]" : "");

   ReverseOnSignal(sig);
   if(CountMine() == 0)
      TryEntry(sig, atr, spreadPts);
}
//+------------------------------------------------------------------+
