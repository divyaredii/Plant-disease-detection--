@echo off
echo ======================================================================
echo   DEPLOYING IMPROVED PLANT DISEASE DETECTION SYSTEM
echo ======================================================================
echo.

cd "Flask Deployed App"

echo [1/4] Backing up original files...
if exist app.py (
    if not exist app_original.py (
        copy app.py app_original.py >nul
        echo   [OK] Backed up app.py
    )
)

if exist templates\submit.html (
    if not exist templates\submit_original.html (
        copy templates\submit.html templates\submit_original.html >nul
        echo   [OK] Backed up submit.html
    )
)

echo.
echo [2/4] Deploying improved files...
if exist app_improved.py (
    copy /Y app_improved.py app.py >nul
    echo   [OK] Deployed app_improved.py -^> app.py
)

if exist templates\submit_improved.html (
    copy /Y templates\submit_improved.html templates\submit.html >nul
    echo   [OK] Deployed submit_improved.html -^> submit.html
)

echo.
echo [3/4] Creating required directories...
if not exist static\uploads (
    mkdir static\uploads
    echo   [OK] Created static\uploads
)

echo.
echo [4/4] Deployment complete!
echo.
echo ======================================================================
echo   IMPROVEMENTS DEPLOYED SUCCESSFULLY!
echo ======================================================================
echo.
echo   New Features:
echo     * Confidence Scores (with color coding)
echo     * Top-3 Predictions
echo     * Grad-CAM Visualization (Explainable AI)
echo     * Performance Metrics
echo     * Enhanced UI/UX
echo     * RESTful API
echo.
echo ======================================================================
echo   NEXT STEPS
echo ======================================================================
echo.
echo   1. Stop the current Flask app (Ctrl+C)
echo   2. Start the improved app:
echo.
echo      python app.py
echo.
echo   3. Open: http://localhost:5000
echo.
echo ======================================================================
echo.

cd ..
pause
