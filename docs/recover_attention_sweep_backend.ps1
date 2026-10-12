param([Parameter(Mandatory=$true)][string]$ResultRoot)
$ErrorActionPreference = 'Stop'
$resultPath = (Resolve-Path -LiteralPath $ResultRoot).Path
if (-not $resultPath.StartsWith('C:\ComfyUI\custom_nodes\FreeFuse-for-sliderlora\docs\attention-step-block-sweep-results\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected result root' }
$record = Get-Content -LiteralPath (Join-Path $resultPath 'backend-restart.before.json') -Raw | ConvertFrom-Json -AsHashtable
$cwd = 'C:\Users\inott\ComfyUI-Installs\ComfyUI'
if (-not (Test-Path -LiteralPath (Join-Path $cwd 'ComfyUI\main.py'))) { throw 'Verified runtime source missing' }
if (@(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).Count) { throw 'Port already has a listener; recovery does not replace it' }
$argv = @($record.argv)
if ($argv[0] -ne '-s' -or $argv[1] -ne 'ComfyUI\main.py') { throw 'Unexpected retained launch arguments' }
$quotedArgs = foreach ($arg in $argv) { '"' + $arg.Replace('"', '\"') + '"' }
$proc = Start-Process -FilePath 'C:\ComfyUI\.venv\Scripts\python.exe' -ArgumentList $quotedArgs -WorkingDirectory $cwd -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resultPath 'backend-recovery.stdout.log') -RedirectStandardError (Join-Path $resultPath 'backend-recovery.stderr.log')
Write-Output "RECOVERY_LAUNCHED $($proc.Id)"
$deadline = [DateTime]::UtcNow.AddMinutes(3)
$notice = [DateTime]::UtcNow
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $after = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/system_stats' -TimeoutSec 3
        $info = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/object_info/Krea2SliderFuseAttentionSweepCollect' -TimeoutSec 3
        if ($null -ne $info.Krea2SliderFuseAttentionSweepCollect) {
            if (($after.system.argv -join [char]0) -ne ($record.before.system.argv -join [char]0)) { throw 'Restored arguments differ' }
            $record.cwd = $cwd
            $record.after = $after
            $record.launched_pid = $proc.Id
            $record.same_runtime_arguments = $true
            $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $resultPath 'backend-recovery.json') -Encoding utf8
            Write-Output 'BACKEND_RECOVERED same arguments; sweep registered'
            exit 0
        }
    } catch { if ($_.Exception.Message -match 'arguments differ') { throw } }
    if ([DateTime]::UtcNow -ge $notice) { Write-Output 'WAITING_FOR_RECOVERY'; $notice = [DateTime]::UtcNow.AddSeconds(20) }
    Start-Sleep -Seconds 2
}
throw 'Recovery timed out; startup logs retained'
