# Define paths for clarity and easier modification
$PythonLibsPath = "C:\Users\bshan\Desktop\PythonLibs"
$PythonExePath = "C:\Users\bshan\AppData\Local\Microsoft\WindowsApps\python3.exe" # Path to your python executable
$ScriptPath = "C:\Github\gemini.py" # Path to your python script

# --- Configuration ---
$CleanInstall = $true # Set to $true to remove the PythonLibsPath directory before installation

# --- Administrator Check ---
Write-Host "Checking for Administrator privileges..."
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "------------------------------------------------------------------"
    Write-Warning "Administrator privileges are required to run this script properly,"
    Write-Warning "especially for the 'CleanInstall' feature (removing directories)."
    Write-Warning "Please re-run this script as an Administrator."
    Write-Warning "To do this, right-click the .ps1 file and select 'Run as administrator',"
    Write-Warning "or open PowerShell as Administrator and then run the script."
    Write-Warning "------------------------------------------------------------------"
    Exit 1 # Exit the script
} else {
    Write-Host "Administrator privileges detected."
}

# 1. Ensure the target directory for Python libraries exists.
Write-Host "Ensuring directory '$PythonLibsPath' exists..."
if ($CleanInstall -and (Test-Path $PythonLibsPath)) {
    Write-Host "Performing a clean install: Removing existing directory '$PythonLibsPath'..."
    try {
        Remove-Item -Path $PythonLibsPath -Recurse -Force -ErrorAction Stop
        Write-Host "Successfully removed '$PythonLibsPath'."
    } catch {
        Write-Error "Failed to remove '$PythonLibsPath'. Error: $($_.Exception.Message)"
        Write-Host "Please ensure no files in this directory are locked by other processes."
        Write-Host "If the error persists even when run as Administrator, check folder permissions or for locked files."
        Exit 1
    }
}

try {
    New-Item -Path $PythonLibsPath -ItemType Directory -Force -ErrorAction Stop | Out-Null
    Write-Host "Directory '$PythonLibsPath' is ready."
} catch {
    Write-Error "Failed to create directory '$PythonLibsPath'. Error: $($_.Exception.Message)"
    Exit 1
}


# 2. Install/upgrade the google-generativeai package into the target directory.
Write-Host "Installing/upgrading 'google-generativeai' into '$PythonLibsPath'..."
$PipArgs = "-m pip install --upgrade --target `"$PythonLibsPath`" google-generativeai"
Write-Host "Executing: $PythonExePath $PipArgs"

$PipProcess = Start-Process -FilePath $PythonExePath -ArgumentList $PipArgs -Wait -PassThru -NoNewWindow

if ($PipProcess.ExitCode -ne 0) {
    Write-Error "pip install failed with exit code $($PipProcess.ExitCode)."
    Write-Host "Please check any output from pip above."
    Exit 1
} else {
    Write-Host "'google-generativeai' installation/upgrade command completed successfully."
}

# 3. Verify installation contents.
Write-Host "Verifying contents of '$PythonLibsPath':"
Get-ChildItem -Path $PythonLibsPath | Select-Object -ExpandProperty Name

$ExpectedGooglePackagePath = Join-Path $PythonLibsPath "google"
if (-not (Test-Path $ExpectedGooglePackagePath)) {
    Write-Warning "CRITICAL: The 'google' directory was NOT found in '$PythonLibsPath' after installation."
    Write-Warning "This means 'google-generativeai' likely did not install correctly into the target path."
    Write-Host "Expected path: $ExpectedGooglePackagePath"
    Exit 1
} else {
    Write-Host "Found 'google' directory in '$PythonLibsPath' as expected."
    Write-Host "Contents of '$ExpectedGooglePackagePath' (should contain 'generativeai'):"
    Get-ChildItem -Path $ExpectedGooglePackagePath | Select-Object -ExpandProperty Name
    if (-not (Test-Path (Join-Path $ExpectedGooglePackagePath "generativeai"))) {
         Write-Warning "CRITICAL: The 'generativeai' subdirectory was NOT found inside '$ExpectedGooglePackagePath'."
         Write-Warning "The 'google-generativeai' library installation seems incomplete."
         Exit 1
    } else {
        Write-Host "'generativeai' subdirectory found. Installation looks promising."
    }
}

# 4. Set the PYTHONPATH environment variable for the current PowerShell session.
Write-Host "Setting PYTHONPATH for this session..."
if ($env:PYTHONPATH) {
    # Prepend our custom path to the existing PYTHONPATH, using the OS-specific path separator.
    # For Windows, the separator is ';'. For Linux/macOS, it's ':'.
    # System.IO.Path.PathSeparator can be used for OS-agnostic separator if needed in more complex scripts.
    $env:PYTHONPATH = "$PythonLibsPath;$($env:PYTHONPATH)"
    Write-Host "Prepended '$PythonLibsPath' to existing PYTHONPATH."
} else {
    $env:PYTHONPATH = $PythonLibsPath # Set new PYTHONPATH if it doesn't exist
    Write-Host "Set new PYTHONPATH to '$PythonLibsPath'."
}
Write-Host "PYTHONPATH is now: $env:PYTHONPATH"


# 5. Perform a direct Python import test.
Write-Host "Attempting a direct import test using Python to verify PYTHONPATH..."
$TestImportCommand = "import sys; print(f'Python sys.path: {sys.path}'); import google.generativeai; print('Test import of google.generativeai successful!')"
# Arguments for python -c. The command string itself needs to be quoted for -c.
$TestImportArgs = "-c `"$TestImportCommand`""

Write-Host "Executing: $PythonExePath $TestImportArgs"
# Redirect output to files for detailed review
$TestImportProcess = Start-Process -FilePath $PythonExePath -ArgumentList $TestImportArgs -Wait -PassThru -NoNewWindow -RedirectStandardOutput "test_import_stdout.txt" -RedirectStandardError "test_import_stderr.txt"

Write-Host ""
Write-Host "--- Test Import STDOUT (from test_import_stdout.txt) ---"
if (Test-Path "test_import_stdout.txt") { Get-Content "test_import_stdout.txt" | Write-Host } else { Write-Host "test_import_stdout.txt not found."}
Write-Host "--- End Test Import STDOUT ---"
Write-Host ""
Write-Host "--- Test Import STDERR (from test_import_stderr.txt) ---"
if (Test-Path "test_import_stderr.txt") { Get-Content "test_import_stderr.txt" | Write-Host } else { Write-Host "test_import_stderr.txt not found."}
Write-Host "--- End Test Import STDERR ---"
Write-Host ""

if ($TestImportProcess.ExitCode -ne 0) {
    Write-Error "Direct Python import test FAILED. Exit code: $($TestImportProcess.ExitCode)."
    Write-Error "This means Python could not find 'google.generativeai' even with PYTHONPATH set to '$PythonLibsPath'."
    Write-Error "Review the sys.path output and STDERR above (in test_import_stdout.txt and test_import_stderr.txt)."
    Write-Error "Ensure '$PythonLibsPath' is listed in sys.path and contains the library correctly."
    Exit 1
} else {
    Write-Host "Direct Python import test SUCCEEDED."
}

# 6. Execute the main Python script.
Write-Host "Executing main Python script: '$ScriptPath'..."
# Ensure script path is quoted if it could have spaces, though current one doesn't.
$MainScriptArgs = "`"$ScriptPath`"" 
$ScriptDirectory = Split-Path -Path $ScriptPath -Parent # Get the directory of the script

Write-Host "Executing: $PythonExePath $MainScriptArgs (Working Directory: $ScriptDirectory)"
# Redirect output to files for detailed review
$MainScriptProcess = Start-Process -FilePath $PythonExePath -ArgumentList $MainScriptArgs -WorkingDirectory $ScriptDirectory -Wait -PassThru -NoNewWindow -RedirectStandardOutput "main_script_stdout.txt" -RedirectStandardError "main_script_stderr.txt"

Write-Host ""
Write-Host "--- Main Script STDOUT (from main_script_stdout.txt) ---"
if (Test-Path "main_script_stdout.txt") { Get-Content "main_script_stdout.txt" | Write-Host } else { Write-Host "main_script_stdout.txt not found."}
Write-Host "--- End Main Script STDOUT ---"
Write-Host ""
Write-Host "--- Main Script STDERR (from main_script_stderr.txt) ---"
if (Test-Path "main_script_stderr.txt") { Get-Content "main_script_stderr.txt" | Write-Host } else { Write-Host "main_script_stderr.txt not found."}
Write-Host "--- End Main Script STDERR ---"
Write-Host ""

if ($MainScriptProcess.ExitCode -ne 0) {
    Write-Warning "The main Python script '$ScriptPath' finished with a non-zero exit code: $($MainScriptProcess.ExitCode)."
    Write-Warning "Check main_script_stdout.txt and main_script_stderr.txt for output and error details from your Python script."
} else {
    Write-Host "Main Python script execution finished successfully."
}

Write-Host "PowerShell script finished."
