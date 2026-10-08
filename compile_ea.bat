@echo off
"C:\Program Files\MetaTrader 5\MetaEditor64.exe" /compile:"%~dp0MambaGoldEA.mq5" /log
echo editor_errorlevel=%errorlevel%
