[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$JobDir,

    # Text in the shape of transcript.txt, corrected by ear or typed from scratch.
    [Parameter(Mandatory)]
    [string]$Reference,

    # Optional bounds (hh:mm:ss or seconds); by default the reference's own span.
    [string]$Start,
    [string]$End
)

. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
Set-Location $repo

$arguments = @(
    'evaluate',
    '--job-dir', (Resolve-RepoPath -Path $JobDir),
    '--reference', (Resolve-RepoPath -Path $Reference)
)
if ($Start) {
    $arguments += @('--start', $Start)
}
if ($End) {
    $arguments += @('--end', $End)
}
Invoke-UvModule @arguments
