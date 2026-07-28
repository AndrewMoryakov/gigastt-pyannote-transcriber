[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string[]]$InputAudio,

    [string]$OutputRoot = '..\gigastt-pyannote-output',
    [string]$Config = 'config\default.yaml',
    [string]$SpeakerMap,
    [ValidateRange(1, 32)]
    [int]$NumSpeakers = 4,
    [switch]$AllowDownmix,
    [switch]$StrictSpeakers,

    # Defaults suit a 16-core CPU; set to your own physical core count.
    [ValidateRange(1, 256)]
    [int]$TorchThreads = 16,
    [ValidateRange(1, 256)]
    [int]$TorchInteropThreads = 1
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Import-DotEnv

$lock = Get-Content -LiteralPath (Join-Path $repo 'tools\tools.lock.json') -Raw -Encoding UTF8 |
    ConvertFrom-Json
$exe = Join-Path $repo "tools\bin\gigastt\$($lock.gigastt.version)\gigastt.exe"
if (-not (Test-Path -LiteralPath $exe)) {
    throw 'Не найден GigaSTT. Запустите .\scripts\download-models.ps1.'
}
$modelRoot = Resolve-RepoPath -Path 'models' -AllowMissing
$gigaModelDir = Join-Path $modelRoot 'gigastt'
$env:GIGASTT_PUNCT_MODEL_DIR = Join-Path $gigaModelDir 'punct'
$env:GIGASTT_VAD_MODEL_DIR = Join-Path $gigaModelDir 'vad'
$env:GIGASTT_OFFLINE = '1'

$outputPath = Resolve-RepoPath -Path $OutputRoot -AllowMissing
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

# Deliberately invoke one complete job at a time. On a CPU-only host, running
# pyannote for multiple recordings concurrently wastes RAM and hurts latency.
foreach ($item in $InputAudio) {
    $inputPath = Resolve-RepoPath -Path $item
    Write-Host "Обрабатываю последовательно: $inputPath"
    $arguments = @(
        'run',
        '--input', $inputPath,
        '--output-root', $outputPath,
        '--config', (Resolve-RepoPath -Path $Config),
        '--gigastt-exe', $exe,
        '--model-dir', $gigaModelDir,
        '--num-speakers', $NumSpeakers.ToString(),
        '--torch-threads', $TorchThreads.ToString(),
        '--torch-interop-threads', $TorchInteropThreads.ToString()
    )
    if ($SpeakerMap) {
        $arguments += @('--speaker-map', (Resolve-RepoPath -Path $SpeakerMap))
    }
    if ($AllowDownmix) {
        $arguments += '--allow-downmix'
    }
    if ($StrictSpeakers) {
        $arguments += '--strict-speakers'
    }
    Invoke-UvModule @arguments
}
