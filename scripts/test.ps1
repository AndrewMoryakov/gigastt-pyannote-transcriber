[CmdletBinding()]
param()

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo
Assert-Command -Name 'uv' -Hint 'Сначала запустите .\scripts\install.ps1.'

& (Join-Path $PSScriptRoot 'check-repo-hygiene.ps1')
if ($LASTEXITCODE -ne 0) {
    throw 'Проверка гигиены репозитория завершилась с ошибкой.'
}

& uv run --python 3.11 ruff check src tests
if ($LASTEXITCODE -ne 0) {
    throw "ruff нашёл замечания (код $LASTEXITCODE)."
}

& uv run --python 3.11 pytest
if ($LASTEXITCODE -ne 0) {
    throw "pytest завершился с кодом $LASTEXITCODE."
}
