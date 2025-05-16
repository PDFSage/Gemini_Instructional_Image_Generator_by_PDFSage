Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$env:GEMINI_DEFAULT_API_KEY = "AIzaSyCKbxb_BxCzi1uCMrEGHTOATqZis6beq50"

if (Test-Path ".\env\Scripts\Activate.ps1") { . ".\env\Scripts\Activate.ps1" | Out-Null }
