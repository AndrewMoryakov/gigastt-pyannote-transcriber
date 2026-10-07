[CmdletBinding()]
param(
    # Do not demand HF_TOKEN, pyannote or torch (see run.ps1 -NoDiarization).
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
if ($NoDiarization) {
    Write-Host '[--] HF_TOKEN не проверяется: диаризация выключена.'
}
elseif (-not $env:HF_TOKEN -or $env:HF_TOKEN -eq 'hf_REPLACE_WITH_YOUR_READ_TOKEN') {
    $problems.Add('Не задан настоящий HF_TOKEN для текущего процесса (или в локальном .env).')
}
else {
    Write-Host '[OK] HF_TOKEN задан (значение не выводится).'
}

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
