# Run the full benchmark as two lanes side by side on disjoint CPUs (Windows), then score.
#
#   powershell -ExecutionPolicy Bypass -File benchmark\run_lanes.ps1 [-Minutes 60] [-Archive run2-all-models] [-Resume]
#
# -Resume skips every dataset/system whose meta.json already says "ok" (after a crash or reboot).
# The script refuses to start when the headless `claude` CLI is not logged in (`claude auth status`).
#
# Lane A: CPUs 0-5, lane B: CPUs 6-11, 14 GB each (see run.py --cpus / --memory-gb). Each lane runs its
# datasets one at a time, AutoGluon then aga. Progress: benchmark\results\raw\_logs\summary.txt.
# score.py rewrites the README benchmark block; with -Archive the results move to
# benchmark\results\raw\_runs\<name>\ and README.md is restored from the copy
# saved just before scoring (README edits made during the run are kept).
param([double]$Minutes = 60, [string]$Archive = "", [switch]$Resume)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
$py = "$repo\.venv\Scripts\python.exe"
$raw = "$repo\benchmark\results\raw"
$logs = "$raw\_logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$summary = "$logs\summary.txt"
Set-Location $repo
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null  # no idle sleep while this runs
function Say($msg) { "$msg $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $summary }
claude auth status *> $null
if ($LASTEXITCODE -ne 0) {  # every aga run would fail in seconds ("OAuth session expired")
    Say "ABORT: claude is not logged in; run 'claude' once and sign in, then start again"
    exit 1
}

$lanes = @(
    @{ Name = "A"; Cpus = "0-5"; Datasets = "football-xg", "ames-housing", "bank-marketing" },
    @{ Name = "B"; Cpus = "6-11"; Datasets = "adult-planted-leak", "california-housing", "bike-sharing" }
)
$jobs = foreach ($lane in $lanes) {
    Start-Job -ArgumentList $lane, $py, $repo, $logs, $summary, $Minutes, $Resume.IsPresent -ScriptBlock {
        param($lane, $py, $repo, $logs, $summary, $Minutes, $Resume)
        Set-Location $repo
        foreach ($d in $lane.Datasets) {
            foreach ($sys in "autogluon", "aga") {
                $meta = "$repo\benchmark\results\raw\$d\$sys\meta.json"
                if ($Resume -and (Test-Path $meta) -and ((Get-Content $meta -Raw | ConvertFrom-Json).status -eq "ok")) {
                    "lane $($lane.Name) $d $sys skipped (already ok) $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $summary
                    continue
                }
                $t0 = Get-Date
                & $py benchmark\run.py --dataset $d --system $sys --minutes $Minutes --cpus $lane.Cpus --memory-gb 14 *> "$logs\$d-$sys.log"
                "lane $($lane.Name) $d $sys exit=$LASTEXITCODE start=$($t0.ToString('s')) min=$([math]::Round(((Get-Date) - $t0).TotalMinutes, 1))" |
                    Out-File -Append -Encoding utf8 $summary
            }
        }
    }
    Start-Sleep 5  # keep the lanes' first summary lines apart
}
Say "lanes started: $($lanes | ForEach-Object { "$($_.Name)=$($_.Datasets -join ',')" })"
$jobs | Wait-Job | Out-Null
$readmeCopy = "$logs\README.before-score.md"
Copy-Item -Force "$repo\README.md" $readmeCopy
& $py benchmark\score.py *> "$logs\score.log"
Say "score exit=$LASTEXITCODE"
if ($Archive) {
    $dest = "$raw\_runs\$Archive"
    New-Item -ItemType Directory -Force $dest | Out-Null
    foreach ($f in "RESULTS.md", "results.csv", "chart.png") {
        if (Test-Path "$repo\benchmark\results\$f") { Move-Item -Force "$repo\benchmark\results\$f" $dest }
    }
    foreach ($lane in $lanes) { foreach ($d in $lane.Datasets) { if (Test-Path "$raw\$d") { Move-Item -Force "$raw\$d" $dest } } }
    Move-Item -Force $logs "$dest\logs"
    Copy-Item -Force "$dest\logs\README.before-score.md" "$repo\README.md"
    "archived to $dest; README.md restored" | Out-File -Encoding utf8 "$dest\logs\summary.txt" -Append
}
