<#
.SYNOPSIS
    Setup helper for Marin on Windows.
.DESCRIPTION
    Checks for prerequisites (uv, Python) and validates platform compatibility.
    Warns about known issues (e.g., vortex-data on Windows).
#>

Write-Host "🌊 Marin Setup Helper" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan

# 1. Check for Python
Write-Host "[1/3] Checking Python..." -NoNewline
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $pyVersion = python --version 2>&1
    Write-Host " OK ($pyVersion)" -ForegroundColor Green
} else {
    Write-Host " MISSING" -ForegroundColor Red
    Write-Host "Error: Python is not found in PATH."
    exit 1
}

# 2. Check for uv
Write-Host "[2/3] Checking uv..." -NoNewline
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    $uvVersion = uv --version 2>&1
    Write-Host " OK ($uvVersion)" -ForegroundColor Green
} else {
    # Check common user install correctness
    $userPath = "$env:APPDATA\Python\Python311\Scripts\uv.exe"
    if (Test-Path $userPath) {
        Write-Host " FOUND (Not in PATH)" -ForegroundColor Yellow
        Write-Host "Warning: 'uv' is installed at '$userPath' but not in your PATH."
        Write-Host "Please add it to your PATH to use 'uv' commands directly."
        # Alias for this session to continue
        New-Alias -Name uv -Value $userPath -Force
    } else {
        Write-Host " MISSING" -ForegroundColor Red
        Write-Host "Error: 'uv' is not installed. Run 'pip install uv'."
        exit 1
    }
}

# 3. Check Platform Compatibility
Write-Host "[3/3] Checking Platform Compatibility..." -NoNewline
$osDetails = Get-CimInstance Win32_OperatingSystem
if ($osDetails.Caption -match "Windows") {
    Write-Host " WARNING" -ForegroundColor Yellow
    Write-Host "`n[!] CRITICAL COMPATIBILITY NOTICE:" -ForegroundColor Red
    Write-Host "    Marin currently depends on 'vortex-data', which does not have pre-built binaries for Windows."
    Write-Host "    You may encounter installation errors."
    Write-Host "    RECOMMENDATION: Use WSL2 (Windows Subsystem for Linux) or Docker for a smooth experience."
    Write-Host "    See: https://learn.microsoft.com/en-us/windows/wsl/install"
} else {
    Write-Host " OK" -ForegroundColor Green
}

Write-Host "`nSetup check complete."
