param([string]$Message = "Update AIGC learning project")

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-GitChecked {
    param([string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: git $($GitArguments -join ' ')" }
}

Invoke-GitChecked -GitArguments @('rev-parse', '--show-toplevel')
Invoke-GitChecked -GitArguments @('add', '--', '.')
$pythonCommand = Get-Command python -ErrorAction Stop
& $pythonCommand.Source -X utf8 (Join-Path $PSScriptRoot 'tools/check_upload.py')
if ($LASTEXITCODE -ne 0) { throw 'Upload checks failed; nothing was pushed.' }

& git diff --cached --quiet
$diffExit = $LASTEXITCODE
if ($diffExit -eq 1) {
    Invoke-GitChecked -GitArguments @('commit', '-m', $Message)
} elseif ($diffExit -ne 0) {
    throw 'Unable to inspect staged changes.'
} else {
    Write-Host 'No new staged changes; checking for an unpublished commit.'
}
Invoke-GitChecked -GitArguments @('push', '-u', 'origin', 'main')
Write-Host 'GitHub synchronization completed.'
