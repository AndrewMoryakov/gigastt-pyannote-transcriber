Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Import-DotEnv {
    param([string]$Path = (Join-Path (Get-RepoRoot) '.env'))

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) {
            continue
        }

        $parts = $trimmed -split '=', 2
        if ($parts.Count -ne 2) {
            throw "Некорректная строка в ${Path}: $line"
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

function Assert-Command {
    param(
        [Parameter(Mandatory)][string]$Name,
        [string]$Hint = ''
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        $message = "Не найдена команда '$Name'."
        if ($Hint) {
            $message += " $Hint"
        }
        throw $message
    }
}

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$AllowMissing
    )

    $repo = Get-RepoRoot
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $repo $Path }
    if ($AllowMissing) {
        return [IO.Path]::GetFullPath($candidate)
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Invoke-UvModule {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)

    Assert-Command -Name 'uv' -Hint 'Сначала запустите .\scripts\install.ps1.'
    & uv run --python 3.11 python -m fourvoices.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "fourvoices завершился с кодом $LASTEXITCODE."
    }
}
