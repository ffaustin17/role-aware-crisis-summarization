param(
    [Parameter(Mandatory = $true)]
    [string]$Base64Path,

    [Parameter(Mandatory = $true)]
    [string]$ZipPath,

    [Parameter(Mandatory = $true)]
    [string]$DestinationPath
)

$resolvedBase64Path = (Resolve-Path -LiteralPath $Base64Path).ProviderPath
$resolvedZipPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ZipPath)
$resolvedDestinationPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DestinationPath)

$base64Text = (Get-Content -Raw -LiteralPath $resolvedBase64Path) -replace '\s', ''
[IO.File]::WriteAllBytes($resolvedZipPath, [Convert]::FromBase64String($base64Text))

Expand-Archive -LiteralPath $resolvedZipPath -DestinationPath $resolvedDestinationPath -Force

Write-Output "Decoded archive: $resolvedZipPath"
Write-Output "Extracted to: $resolvedDestinationPath"
