param([Parameter(Mandatory=$true)][string]$ResultRoot)
$ErrorActionPreference = 'Stop'
$resultPath = (Resolve-Path -LiteralPath $ResultRoot).Path
$workspacePath = 'C:\ComfyUI\custom_nodes\FreeFuse-for-sliderlora\docs\attention-step-block-sweep-results\'
if (-not $resultPath.StartsWith($workspacePath, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected result root' }
$baseUrl = 'http://127.0.0.1:8000'
$before = Invoke-RestMethod -Uri "$baseUrl/system_stats"
$queue = Invoke-RestMethod -Uri "$baseUrl/queue"
if ($queue.queue_running.Count -or $queue.queue_pending.Count) { throw 'Refusing to restart a busy ComfyUI queue' }
$listeners = @(Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique)
if ($listeners.Count -ne 1) { throw 'Expected one exact ComfyUI listener' }
$server = Get-CimInstance Win32_Process -Filter "ProcessId = $($listeners[0])"
if ($server.Name -ne 'python.exe' -or $server.CommandLine -notmatch 'ComfyUI\\main.py' -or $server.CommandLine -notmatch '--port["\s]+8000') { throw 'Listener is not the expected backend' }
$venvPython = 'C:\ComfyUI\.venv\Scripts\python.exe'
$wrapper = Get-CimInstance Win32_Process -Filter "ProcessId = $($server.ParentProcessId)"
$targets = @($server)
if ($wrapper.ExecutablePath -eq $venvPython -and $wrapper.CommandLine -match 'ComfyUI\\main.py') { $targets += $wrapper }
$argv = @('-s') + @($before.system.argv)
if ($argv[1] -ne 'ComfyUI\main.py' -or $before.system.argv -notcontains '--port') { throw 'Unexpected runtime argv' }
$runtimeWorkingDirectory = 'C:\Users\inott\ComfyUI-Installs\ComfyUI'
if (-not (Test-Path -LiteralPath (Join-Path $runtimeWorkingDirectory 'ComfyUI\main.py'))) { throw 'Runtime source path missing' }
$record = @{before=$before; processes=$targets | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine; argv=$argv; cwd=$runtimeWorkingDirectory}
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $resultPath 'backend-restart.before.json') -Encoding utf8
foreach ($target in $targets) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($target.ProcessId)"
    if ($null -eq $current) { continue }
    if ($current.ExecutablePath -ne $target.ExecutablePath -or $current.CommandLine -ne $target.CommandLine) { throw 'Backend process identity changed' }
    Stop-Process -Id $target.ProcessId -Force
}
$quotedArgs = foreach ($arg in $argv) { '"' + $arg.Replace('"', '\"') + '"' }
$launched = Start-Process -FilePath $venvPython -ArgumentList $quotedArgs -WorkingDirectory $runtimeWorkingDirectory -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resultPath 'backend.stdout.log') -RedirectStandardError (Join-Path $resultPath 'backend.stderr.log')
Write-Output "LAUNCHED_BACKEND $($launched.Id)"
$deadline = [DateTime]::UtcNow.AddMinutes(3)
$notice = [DateTime]::UtcNow
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $after = Invoke-RestMethod -Uri "$baseUrl/system_stats" -TimeoutSec 3
        $info = Invoke-RestMethod -Uri "$baseUrl/object_info/Krea2SliderFuseAttentionSweepCollect" -TimeoutSec 3
        if ($null -ne $info.Krea2SliderFuseAttentionSweepCollect) {
            if (($after.system.argv -join [char]0) -ne ($before.system.argv -join [char]0)) { throw 'Restored backend arguments differ' }
            $record.after = $after
            $record.launched_pid = $launched.Id
            $record.same_runtime_arguments = $true
            $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $resultPath 'backend-restart.json') -Encoding utf8
            Write-Output 'BACKEND_READY same runtime arguments; sweep node registered'
            exit 0
        }
    } catch {
        if ($_.Exception.Message -match 'arguments differ') { throw }
    }
    if ([DateTime]::UtcNow -ge $notice) { Write-Output 'WAITING_FOR_BACKEND'; $notice = [DateTime]::UtcNow.AddSeconds(20) }
    Start-Sleep -Seconds 2
}
throw 'Backend did not recover within three minutes; retained startup logs'
