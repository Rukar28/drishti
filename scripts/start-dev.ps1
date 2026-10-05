# Start both local servers, capture logs, and stop only these child processes on exit.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nodePath = (Get-Command node.exe).Source
$vitePath = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
if (!(Test-Path -LiteralPath $pythonPath) -or !(Test-Path -LiteralPath $vitePath)) {
    throw 'Install backend dependencies and run npm ci inside frontend first.'
}
$logDirectory = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$backendProcess = $null
$frontendProcess = $null
try {
    $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'backend.out.log') -RedirectStandardError (Join-Path $logDirectory 'backend.err.log')
    $frontendProcess = Start-Process -FilePath $nodePath -ArgumentList @($vitePath,'--host','127.0.0.1') -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'frontend.out.log') -RedirectStandardError (Join-Path $logDirectory 'frontend.err.log')
    Write-Host 'VisionMate: http://127.0.0.1:3000 — Ctrl+C stops both servers. Logs: logs/'
    while (!$backendProcess.HasExited -and !$frontendProcess.HasExited) { Start-Sleep -Seconds 1 }
    throw 'A server stopped. Check logs/ for details and ensure ports 3000/8000 are free.'
} finally {
    foreach ($childProcess in @($backendProcess,$frontendProcess)) {
        if ($null -ne $childProcess -and !$childProcess.HasExited) { Stop-Process -Id $childProcess.Id }
    }
}
