$ErrorActionPreference = "Stop"

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$repoName = "football-complete-v3-training"
$branchName = "codex/football-complete-v3-training"
$baseBranch = "main"
$prTitle = "Add Football-Data residual football score training system"
$prBodyPath = Join-Path $root "models\football_complete_v3\PR_DESCRIPTION.md"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is not installed or is not on PATH."
}

function Test-GhAuth {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    gh auth status *> $null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return $exitCode -eq 0
}

function Test-GhRepoExists {
    param([string] $Repository)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    gh repo view $Repository *> $null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return $exitCode -eq 0
}

if (-not (Test-GhAuth)) {
    gh auth login --hostname github.com --web --git-protocol https
}

if ((git status --porcelain).Trim()) {
    throw "Working tree is not clean. Commit or discard changes before creating the PR."
}

$owner = (gh api user --jq ".login").Trim()
$remoteUrl = (git config --get remote.origin.url)

if (-not $remoteUrl) {
    $repoFullName = "$owner/$repoName"
    if (-not (Test-GhRepoExists -Repository $repoFullName)) {
        gh repo create $repoFullName --private --description "Football-Data residual exact-score training system" --disable-wiki
    }
    git remote add origin "https://github.com/$repoFullName.git"
} else {
    $repoFullName = (gh repo view --json nameWithOwner --jq ".nameWithOwner").Trim()
}

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne $branchName) {
    git switch $branchName
}

$remoteBaseExists = (git ls-remote --heads origin $baseBranch).Trim()
if (-not $remoteBaseExists) {
    $emptyTree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    $baseCommit = (git commit-tree $emptyTree -m "Initialize empty PR base").Trim()
    git push origin "${baseCommit}:refs/heads/$baseBranch"
}

git push --set-upstream origin $branchName

$existingPr = (gh pr list --head $branchName --base $baseBranch --json number --jq ".[0].number")
if ($existingPr) {
    gh pr edit $existingPr --title $prTitle --body-file $prBodyPath --base $baseBranch
    gh pr view $existingPr --web
    gh pr view $existingPr --json url --jq ".url"
} else {
    gh pr create --base $baseBranch --head $branchName --title $prTitle --body-file $prBodyPath --web
}
