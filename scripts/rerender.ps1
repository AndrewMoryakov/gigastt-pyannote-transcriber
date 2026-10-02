[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$JobDir,

    [Parameter(Mandatory)]
    [string]$SpeakerMap
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo

Invoke-UvModule render `
    --job-dir (Resolve-RepoPath -Path $JobDir) `
    --speaker-map (Resolve-RepoPath -Path $SpeakerMap)
