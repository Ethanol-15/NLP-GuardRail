@echo off
setlocal

:: The Python script, The virtual environment folder name and current venv folder---
set "SCRIPT_FILE=Scripts\FrontEnd.py"
set "VENV_DIR=venv"
set "CURRENT_DIR=%CD%"

:SEARCH_VENV
    if exist "%CURRENT_DIR%\%VENV_DIR%" (
        set "VENV_PATH=%CURRENT_DIR%\%VENV_DIR%"
        goto VENV_FOUND
    )
	
    :: Move up one directory
    set "CURRENT_DIR=%CURRENT_DIR%\.."
    
    :: Check if we've reached the root directory (or if the path is too short)
    if "%CURRENT_DIR%" equ "%CURRENT_DIR%\" (
        goto VENV_NOT_FOUND
    )
    :: Normalize path and loop again
    for /f "delims=" %%i in ("%CURRENT_DIR%") do set "CURRENT_DIR=%%~fi"
    if /i "%CURRENT_DIR%" equ "%~dp0" (
        goto VENV_NOT_FOUND
    )
    goto SEARCH_VENV

:VENV_FOUND
    echo Found virtual environment: "%VENV_PATH%"
    goto RUN_SCRIPT

:VENV_NOT_FOUND
    echo ERROR: Could not find the 'venv' directory in the current path or any parent directory.
    echo Please make sure you are running this script from within your project folder.
    goto END

:: --- Activate VENV and Run the Streamlit Script ---
:RUN_SCRIPT
	:: Run Sanity Checks
    :: Check if the venv activation script exists
    if not exist "%VENV_PATH%\Scripts\activate.bat" (
        echo ERROR: Virtual environment activation script not found at "%VENV_PATH%\Scripts\activate.bat"
        goto END
    )

    :: Check if the Streamlit Python script exists
    if not exist "%~dp0%SCRIPT_FILE%" (
        echo ERROR: Python script not found at "%~dp0%SCRIPT_FILE%"
        goto END
    )
	
    echo Activating virtual environment and starting Streamlit app...

    :: The 'call' command is used to execute the batch file (activate.bat)
    :: The '&' is a command separator, running the next command after activation
    call "%VENV_PATH%\Scripts\activate.bat" && (
        :: Run the script using the venv's streamlit executable
        streamlit run "%~dp0%SCRIPT_FILE%"
    )

    :: Note: 'deactivate' is not explicitly needed here as the environment is 
    :: only active within this script's execution context.

:END
    endlocal
    pause