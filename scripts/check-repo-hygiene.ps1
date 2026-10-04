[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Assert-Command -Name 'git'

$candidates = @(& git ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw 'git ls-files завершился с ошибкой.'
}

$errors = [Collections.Generic.List[string]]::new()
$audioExtensions = @('.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.wma')
$forbiddenPrefixes = @('media/', 'transcripts/', 'secrets/', 'models/', 'tools/bin/', 'tools/downloads/')

foreach ($relative in $candidates) {
    $normalized = $relative.Replace('\', '/')
    $extension = [IO.Path]::GetExtension($normalized).ToLowerInvariant()
    if ($audioExtensions -contains $extension) {
        $errors.Add("Аудиофайл не должен попадать в Git: $normalized")
    }
    foreach ($prefix in $forbiddenPrefixes) {
        if ($normalized.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            $errors.Add("Локальный/generated файл не должен попадать в Git: $normalized")
            break
        }
    }
    if ($normalized -eq '.env' -or ($normalized.StartsWith('.env.') -and $normalized -ne '.env.example')) {
        $errors.Add("Файл окружения не должен попадать в Git: $normalized")
    }
}

$secretPattern = 'hf_[A-Za-z0-9]{20,}'
foreach ($relative in $candidates) {
    $fullPath = Join-Path $repo $relative
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        continue
    }
    try {
        $content = Get-Content -LiteralPath $fullPath -Raw -Encoding UTF8 -ErrorAction Stop
    }
    catch {
        continue
    }
    if ($content -match $secretPattern) {
        $errors.Add("Похожий на настоящий Hugging Face token найден в: $relative")
    }
}

if ($errors.Count -gt 0) {
    $errors | Sort-Object -Unique | ForEach-Object { Write-Error $_ }
    throw "Нарушена гигиена репозитория: $($errors.Count) проблем."
}

Write-Host '[OK] Аудио, модели, транскрипты и секреты не попадут в Git.'
