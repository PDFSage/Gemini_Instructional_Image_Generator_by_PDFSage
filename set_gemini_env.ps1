Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$env:GEMINI_DEFAULT_API_KEY = "HERE"

if (Test-Path ".\env\Scripts\Activate.ps1") { . ".\env\Scripts\Activate.ps1" | Out-Null }
