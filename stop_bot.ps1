<#
    Stops every running copy of this project's Telegram bot.

    Telegram allows exactly one getUpdates consumer per bot token. If two
    copies run at once they cancel each other's long-poll requests and both
    answer with "409 Conflict: terminated by other getUpdates request".

    Processes are matched by command line (project path + src\bot.py), not by
    window title, so this also works when the console window is hidden,
    minimised or renamed.

    Usage:
        powershell -NoProfile -ExecutionPolicy Bypass -File stop_bot.ps1
#>
$ErrorActionPreference = 'SilentlyContinue'

$needle = Join-Path $PSScriptRoot 'src\bot.py'

$targets = @(
    Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$needle*" }
)

if ($targets.Count -eq 0) {
    Write-Output 'No running bot instances found.'
    exit 0
}

foreach ($process in $targets) {
    Write-Output "Stopping PID $($process.ProcessId)"
    Stop-Process -Id $process.ProcessId -Force
}

Write-Output "Stopped $($targets.Count) process(es)."
Start-Sleep -Milliseconds 800
exit 0