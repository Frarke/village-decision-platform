$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

try {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[1/3] 正在检测Python并创建独立运行环境……" -ForegroundColor Cyan
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if ($Python) {
            & python -m venv .venv
        } else {
            $Launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($Launcher) {
                & py -3 -m venv .venv
            } else {
                throw "未检测到Python。请安装64位Python 3.10至3.12，并勾选 Add Python to PATH。"
            }
        }
    }
    Write-Host "[2/3] 正在检查平台依赖……" -ForegroundColor Cyan
    & $VenvPython -c "import streamlit,plotly,pandas,numpy,pytest" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "首次启动需要安装依赖，请保持网络可用。" -ForegroundColor Yellow
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "依赖安装未完成，请检查网络或代理设置。" }
    }
    Write-Host "[3/3] 正在启动平台：http://localhost:8501" -ForegroundColor Green
    & $VenvPython -m streamlit run app.py
} catch {
    Write-Host "`n[启动失败] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请保留本窗口中的错误信息用于排查。" -ForegroundColor Yellow
    Read-Host "按回车键关闭"
    exit 1
}
