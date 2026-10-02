[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Import-DotEnv

# A shell that was opened before ffmpeg was installed has a stale PATH; re-read
# the machine and user values so the UI finds it without restarting anything.
$env:Path = @(
    $env:Path,
    [Environment]::GetEnvironmentVariable('Path', 'User'),
    [Environment]::GetEnvironmentVariable('Path', 'Machine')
) -join ';'

Assert-Command -Name 'uv' -Hint 'Сначала запустите .\scripts\install.ps1.'

$arguments = @('run', '--python', '3.11', 'python', '-m', 'fourvoices.webui', '--port', $Port.ToString())
if ($NoBrowser) {
    $arguments += '--no-browser'
}
& uv @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Интерфейс завершился с кодом $LASTEXITCODE."
}
