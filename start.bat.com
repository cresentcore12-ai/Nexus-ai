@echo off
echo ==============================================
echo   NEXUS AI ULTIMATE - Revolutionary Edition
echo   Made in Uttarakhand, India
echo ==============================================
echo.
echo Revolutionary Features Loading...
echo.
python server.py --load-models
pause
