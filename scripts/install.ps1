[CmdletBinding()]
param(
    [switch]$InstallFfmpeg,
    [switch]$SkipSync
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'Нужна 64-разрядная Windows.'
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Assert-Command -Name 'winget' -Hint 'Установите uv вручную: https://docs.astral.sh/uv/getting-started/installation/'
    Write-Host 'Устанавливаю uv через winget...'
    & winget install --id astral-sh.uv --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw 'winget не смог установить uv. Установите uv вручную и повторите команду.'
    }

    $uvCandidates = @(
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'),
        (Join-Path $HOME '.local\bin')
    )
    foreach ($candidate in $uvCandidates) {
        if ((Test-Path -LiteralPath $candidate) -and ($env:Path -notlike "*$candidate*")) {
            $env:Path = "$candidate;$env:Path"
        }
    }
}
Assert-Command -Name 'uv'

Write-Host 'Устанавливаю управляемый Python 3.11...'
& uv python install 3.11
if ($LASTEXITCODE -ne 0) {
    throw 'Не удалось установить Python 3.11 через uv.'
}

if (-not $SkipSync) {
    Write-Host 'Создаю .venv и устанавливаю зависимости (CPU PyTorch)...'
    & uv sync --python 3.11
    if ($LASTEXITCODE -ne 0) {
        throw 'uv sync завершился с ошибкой.'
    }
}

if ($InstallFfmpeg -and -not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Assert-Command -Name 'winget' -Hint 'Или установите ffmpeg вручную и добавьте его в PATH.'
    Write-Host 'Устанавливаю ffmpeg через winget...'
    & winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw 'winget не смог установить ffmpeg.'
    }
}

foreach ($directory in @('media', 'transcripts', 'models', 'tools\bin', 'tools\downloads')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $repo $directory) | Out-Null
}

Write-Host ''
Write-Host 'Среда Python готова.'
Write-Host 'Далее: задайте HF_TOKEN только для текущего процесса и запустите .\scripts\download-models.ps1'
