[CmdletBinding()]
param(
    [switch]$SkipGigaStt,
    [switch]$SkipPyannote,
    [switch]$Force
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Import-DotEnv

$lockPath = Join-Path $repo 'tools\tools.lock.json'
$lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
$tool = $lock.gigastt
$downloads = Join-Path $repo 'tools\downloads'
$binRoot = Join-Path $repo "tools\bin\gigastt\$($tool.version)"
$archive = Join-Path $downloads $tool.asset
$exe = Join-Path $binRoot 'gigastt.exe'
New-Item -ItemType Directory -Force -Path $downloads | Out-Null

if (-not $SkipGigaStt) {
    $mustDownload = $Force -or -not (Test-Path -LiteralPath $archive)
    if (-not $mustDownload) {
        $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        $mustDownload = $actual -ne $tool.sha256
        if ($mustDownload) {
            Write-Warning 'Локальный архив GigaSTT не прошёл SHA-256 и будет скачан заново.'
            Remove-Item -LiteralPath $archive -Force
        }
    }

    if ($mustDownload) {
        Write-Host "Скачиваю GigaSTT $($tool.version)..."
        $partial = "$archive.partial"
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        $downloaded = $false
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                Invoke-WebRequest -Uri $tool.url -OutFile $partial -UseBasicParsing
                $downloaded = $true
                break
            }
            catch {
                Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
                if ($attempt -eq 3) {
                    throw
                }
                Write-Warning "Скачивание прервалось (попытка $attempt из 3). Повтор через 3 секунды."
                Start-Sleep -Seconds 3
            }
        }
        if (-not $downloaded) {
            throw 'Не удалось скачать архив GigaSTT.'
        }
        $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $tool.sha256) {
            Remove-Item -LiteralPath $partial -Force
            throw "SHA-256 GigaSTT не совпал. Ожидался $($tool.sha256), получен $actual."
        }
        Move-Item -LiteralPath $partial -Destination $archive -Force
    }

    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $tool.sha256) {
        throw "SHA-256 GigaSTT не совпал. Ожидался $($tool.sha256), получен $actual."
    }
    Write-Host "SHA-256 подтверждён: $actual"

    if ($Force -or -not (Test-Path -LiteralPath $exe)) {
        Assert-Command -Name 'tar' -Hint 'В Windows 10/11 tar входит в комплект ОС.'
        $stage = Join-Path $downloads ("gigastt-stage-" + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $stage | Out-Null
        try {
            & tar -xzf $archive -C $stage
            if ($LASTEXITCODE -ne 0) {
                throw 'Не удалось распаковать архив GigaSTT.'
            }
            $found = Get-ChildItem -LiteralPath $stage -Recurse -Filter 'gigastt.exe' -File |
                Select-Object -First 1
            if (-not $found) {
                throw 'В проверенном архиве не найден gigastt.exe.'
            }
            New-Item -ItemType Directory -Force -Path $binRoot | Out-Null
            Get-ChildItem -LiteralPath $found.Directory.FullName -Force | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $binRoot -Recurse -Force
            }
        }
        finally {
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host 'Скачиваю GigaAM RNNT (INT8) в models\gigastt...'
    $gigaModelDir = Join-Path $repo 'models\gigastt'
    New-Item -ItemType Directory -Force -Path $gigaModelDir | Out-Null
    $modelReady = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        & $exe download --model-dir $gigaModelDir --model-variant rnnt --skip-diarization
        if ($LASTEXITCODE -eq 0) {
            $modelReady = $true
            break
        }
        if ($attempt -lt 3) {
            Write-Warning "gigastt download прервался (попытка $attempt из 3). Повтор через 3 секунды."
            Start-Sleep -Seconds 3
        }
    }
    if (-not $modelReady) {
        throw 'gigastt download не завершился после трёх попыток.'
    }

    # `gigastt download` fetches the recognition model, but punctuation and VAD
    # are lazy assets. Run a one-second local probe so the later evidence run
    # cannot unexpectedly require network access.
    $punctDir = Join-Path $gigaModelDir 'punct'
    $vadDir = Join-Path $gigaModelDir 'vad'
    New-Item -ItemType Directory -Force -Path $punctDir, $vadDir | Out-Null
    $probeWav = Join-Path $env:TEMP ("fourvoices-probe-" + [Guid]::NewGuid().ToString('N') + '.wav')
    $probeJson = "$probeWav.json"
    $sampleRate = 16000
    $dataLength = $sampleRate * 2
    $stream = [IO.File]::Create($probeWav)
    $writer = [IO.BinaryWriter]::new($stream)
    try {
        $writer.Write([Text.Encoding]::ASCII.GetBytes('RIFF'))
        $writer.Write([int](36 + $dataLength))
        $writer.Write([Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
        $writer.Write([int]16)
        $writer.Write([int16]1)
        $writer.Write([int16]1)
        $writer.Write([int]$sampleRate)
        $writer.Write([int]($sampleRate * 2))
        $writer.Write([int16]2)
        $writer.Write([int16]16)
        $writer.Write([Text.Encoding]::ASCII.GetBytes('data'))
        $writer.Write([int]$dataLength)
        $writer.Write([byte[]]::new($dataLength))
    }
    finally {
        $writer.Dispose()
        $stream.Dispose()
    }
    try {
        Write-Host 'Предзагружаю модели пунктуации и VAD...'
        & $exe transcribe $probeWav `
            --model-dir $gigaModelDir `
            --model-variant rnnt `
            --punct-model-dir $punctDir `
            --vad-model-dir $vadDir `
            --punctuation on `
            --itn on `
            --vad `
            --format json `
            --output $probeJson
        if ($LASTEXITCODE -ne 0) {
            throw "Предзагрузка punctuation/VAD завершилась с кодом $LASTEXITCODE."
        }
    }
    finally {
        Remove-Item -LiteralPath $probeWav, $probeJson -Force -ErrorAction SilentlyContinue
    }
}

if (-not $SkipPyannote) {
    Assert-Command -Name 'uv' -Hint 'Сначала запустите .\scripts\install.ps1.'
    if (-not $env:HF_TOKEN -or $env:HF_TOKEN -eq 'hf_REPLACE_WITH_YOUR_READ_TOKEN') {
        throw 'Не задан настоящий HF_TOKEN. Сначала примите условия модели, создайте read-токен Hugging Face и задайте его только для текущего процесса.'
    }

    $pyannoteDir = Join-Path $repo 'models\pyannote'
    New-Item -ItemType Directory -Force -Path $pyannoteDir | Out-Null
    Invoke-UvModule diarize-preload `
        --model $lock.pyannote.model `
        --revision $lock.pyannote.revision `
        --model-dir $pyannoteDir
}

Write-Host 'Все выбранные модели готовы.'
