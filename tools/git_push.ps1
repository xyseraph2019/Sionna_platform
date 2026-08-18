# git_push.ps1 - Push the current branch using the locally stored credential.
#
# Reads the token from D:\Platform\.git-credentials.local (created by the user's
# first `git push` under the repo-local store helper; gitignored) and injects it
# into the origin URL. The token is never printed: the URL is assembled in
# memory and any occurrence of the token in git's output is masked.
#
# Usage:  powershell -File tools\git_push.ps1 [branch]   (default: current branch)
#         powershell -File tools\git_push.ps1 -DryRun    (no remote changes)
param(
    [string]$Branch = "",
    [switch]$DryRun
)

# Note: do NOT set $ErrorActionPreference='Stop' here - git writes benign cygwin
# "signal pipe" warnings to stderr in sandboxed environments, and Stop would turn
# those into terminating errors. We rely on explicit $LASTEXITCODE checks instead.

$credFile = Join-Path $PSScriptRoot "..\.git-credentials.local"
if (-not (Test-Path $credFile)) {
    Write-Host "Credential file not found: $credFile"
    Write-Host "Run once in your terminal:  git push origin main   (then retry)"
    exit 1
}

$raw = [IO.File]::ReadAllText((Resolve-Path $credFile))
$m = [regex]::Match($raw, '^https://([^:]+):([^@]+)@([^\s]+)$')
if (-not $m.Success) {
    Write-Host "Credential file format invalid (expected https://user:token@host): $credFile"
    exit 1
}
$u = $m.Groups[1].Value
$p = $m.Groups[2].Value
Write-Host "Pushing as user '$u' (token masked) ..."

$origin = (git remote get-url origin).Trim()
if ($origin -notmatch '^https://') {
    Write-Host "origin is not an https URL: $origin"
    exit 1
}
$url = $origin -replace '^https://', "https://$u`:$p@"

if (-not $Branch) {
    $Branch = (git branch --show-current).Trim()
}
if (-not $Branch) {
    Write-Host "Could not determine current branch."
    exit 1
}

if ($DryRun) {
    $out = git push --dry-run $url $Branch 2>&1 | Out-String
} else {
    $out = git push $url $Branch 2>&1 | Out-String
}
$code = $LASTEXITCODE
Write-Host ($out.Replace($p, "***"))
exit $code
