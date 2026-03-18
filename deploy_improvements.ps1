# Deploy Improved Plant Disease Detection System
# Run this script to deploy all improvements

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  DEPLOYING IMPROVED PLANT DISEASE DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if we're in the right directory
if (!(Test-Path "Flask Deployed App")) {
    Write-Host "ERROR: Please run this script from the Plant-Disease-Detection directory" -ForegroundColor Red
    exit 1
}

Set-Location "Flask Deployed App"

# Step 2: Backup original files
Write-Host "[1/5] Backing up original files..." -ForegroundColor Yellow
if (Test-Path "app.py") {
    if (!(Test-Path "app_original.py")) {
        Copy-Item "app.py" "app_original.py"
        Write-Host "  ✓ Backed up app.py to app_original.py" -ForegroundColor Green
    } else {
        Write-Host "  ℹ app_original.py already exists, skipping backup" -ForegroundColor Gray
    }
}

if (Test-Path "templates\submit.html") {
    if (!(Test-Path "templates\submit_original.html")) {
        Copy-Item "templates\submit.html" "templates\submit_original.html"
        Write-Host "  ✓ Backed up submit.html to submit_original.html" -ForegroundColor Green
    } else {
        Write-Host "  ℹ submit_original.html already exists, skipping backup" -ForegroundColor Gray
    }
}

# Step 3: Deploy improved files
Write-Host ""
Write-Host "[2/5] Deploying improved files..." -ForegroundColor Yellow

if (Test-Path "app_improved.py") {
    Copy-Item "app_improved.py" "app.py" -Force
    Write-Host "  ✓ Deployed app_improved.py → app.py" -ForegroundColor Green
} else {
    Write-Host "  ✗ app_improved.py not found!" -ForegroundColor Red
}

if (Test-Path "templates\submit_improved.html") {
    Copy-Item "templates\submit_improved.html" "templates\submit.html" -Force
    Write-Host "  ✓ Deployed submit_improved.html → submit.html" -ForegroundColor Green
} else {
    Write-Host "  ✗ submit_improved.html not found!" -ForegroundColor Red
}

# Step 4: Check dependencies
Write-Host ""
Write-Host "[3/5] Checking dependencies..." -ForegroundColor Yellow

$packages = @("grad-cam", "opencv-python", "matplotlib", "scikit-learn")
$missing = @()

foreach ($package in $packages) {
    $result = pip show $package 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $package installed" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $package NOT installed" -ForegroundColor Red
        $missing += $package
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "  Installing missing packages..." -ForegroundColor Yellow
    pip install $missing
}

# Step 5: Create uploads directory
Write-Host ""
Write-Host "[4/5] Creating required directories..." -ForegroundColor Yellow

if (!(Test-Path "static\uploads")) {
    New-Item -ItemType Directory -Path "static\uploads" -Force | Out-Null
    Write-Host "  ✓ Created static\uploads directory" -ForegroundColor Green
} else {
    Write-Host "  ℹ static\uploads already exists" -ForegroundColor Gray
}

# Step 6: Summary
Write-Host ""
Write-Host "[5/5] Deployment Summary" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ✅ IMPROVEMENTS DEPLOYED SUCCESSFULLY!" -ForegroundColor Green
Write-Host ""
Write-Host "  New Features:" -ForegroundColor Cyan
Write-Host "    • Confidence Scores (with color coding)" -ForegroundColor White
Write-Host "    • Top-3 Predictions" -ForegroundColor White
Write-Host "    • Grad-CAM Visualization (Explainable AI)" -ForegroundColor White
Write-Host "    • Performance Metrics (inference time)" -ForegroundColor White
Write-Host "    • Enhanced UI/UX with icons" -ForegroundColor White
Write-Host "    • RESTful API endpoint" -ForegroundColor White
Write-Host ""

# Step 7: Instructions
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Stop the current Flask app (Ctrl+C in the other terminal)" -ForegroundColor Yellow
Write-Host "  2. Start the improved app:" -ForegroundColor Yellow
Write-Host ""
Write-Host "     python app.py" -ForegroundColor Green
Write-Host ""
Write-Host "  3. Open your browser:" -ForegroundColor Yellow
Write-Host ""
Write-Host "     http://localhost:5000" -ForegroundColor Green
Write-Host ""
Write-Host "  4. Upload a plant image and see the improvements!" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Return to original directory
Set-Location ..

Write-Host "Deployment complete! 🚀" -ForegroundColor Green
Write-Host ""
