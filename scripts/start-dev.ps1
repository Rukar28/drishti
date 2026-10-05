param(
    [ValidateSet('configured','webcam','esp32','mock')][string]$Camera = 'configured',
    [ValidateSet('configured','browser','mock')][string]$GPS = 'configured',
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000
)
# Start both local servers, capture logs, and stop only these child processes on exit.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nodePath = (Get-Command node.exe).Source
$vitePath = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
if (!(Test-Path -LiteralPath $pythonPath) -or !(Test-Path -LiteralPath $vitePath)) {
    throw 'Install backend dependencies and run npm ci inside frontend first.'
}
$previousCamera = $env:CAMERA_SOURCE
$previousGPS = $env:GPS_SOURCE
$previousProxy = $env:BACKEND_PROXY_URL
if ($Camera -ne 'configured') { $env:CAMERA_SOURCE = $Camera }
if ($GPS -ne 'configured') { $env:GPS_SOURCE = $GPS }
$env:BACKEND_PROXY_URL = "http://127.0.0.1:$BackendPort"
$logDirectory = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$backendProcess = $null
$frontendProcess = $null
try {
    $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',"$BackendPort") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'backend.out.log') -RedirectStandardError (Join-Path $logDirectory 'backend.err.log')
    $frontendProcess = Start-Process -FilePath $nodePath -ArgumentList @(('"' + $vitePath + '"'),'--host','127.0.0.1','--port',"$FrontendPort",'--strictPort') -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'frontend.out.log') -RedirectStandardError (Join-Path $logDirectory 'frontend.err.log')
    Write-Host "VisionMate: http://127.0.0.1:$FrontendPort - Ctrl+C stops both servers. Logs: logs/"
    while (!$backendProcess.HasExited -and !$frontendProcess.HasExited) { Start-Sleep -Seconds 1 }
    throw "A server stopped. Check logs/ for details and ensure ports $FrontendPort/$BackendPort are free."
} finally {
    $env:CAMERA_SOURCE = $previousCamera
    $env:GPS_SOURCE = $previousGPS
    $env:BACKEND_PROXY_URL = $previousProxy
    foreach ($childProcess in @($backendProcess,$frontendProcess)) {
        if ($null -ne $childProcess -and !$childProcess.HasExited) { Stop-Process -Id $childProcess.Id }
    }
}
