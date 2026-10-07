[CmdletBinding()]
param(
    # Do not demand the pyannote model or its packages even if the configuration
    # leaves diarization on (see run.ps1 -NoDiarization).
    [switch]$NoDiarization
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Import-DotEnv

$lock = Get-Content -LiteralPath (Join-Path $repo 'tools\tools.lock.json') -Raw -Encoding UTF8 |
    ConvertFrom-Json
$exe = Join-Path $repo "tools\bin\gigastt\$($lock.gigastt.version)\gigastt.exe"

$problems = [Collections.Generic.List[string]]::new()
foreach ($command in @('uv', 'ffmpeg', 'ffprobe')) {
    if (Get-Command $command -ErrorAction SilentlyContinue) {
        Write-Host "[OK] $command"
    }
    else {
        $problems.Add("Не найдена команда $command.")
    }
}
if (Test-Path -LiteralPath $exe) {
    Write-Host "[OK] GigaSTT: $exe"
}
else {
    $problems.Add('Не найден gigastt.exe. Запустите .\scripts\download-models.ps1.')
}
# HF_TOKEN is checked by the CLI doctor below, which knows whether diarization is
# on (the configuration or -NoDiarization may turn it off).

if ($problems.Count -gt 0) {
    $problems | ForEach-Object { Write-Error $_ }
    throw "Doctor обнаружил проблем: $($problems.Count)."
}

$doctorArguments = @(
    'doctor',
    '--config', (Join-Path $repo 'config\default.yaml'),
    '--gigastt-exe', $exe,
    '--model-dir', (Join-Path $repo 'models\gigastt')
)
if ($NoDiarization) {
    $doctorArguments += '--no-diarization'
}
Invoke-UvModule @doctorArguments
