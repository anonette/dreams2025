@echo off
echo 🔑 API KEY CHECKER
echo ==================
echo.

if defined OPENAI_API_KEY (
    echo ✅ OpenAI API key is set!
    echo Key ends with: ...%OPENAI_API_KEY:~-4%
    echo.
    echo Ready to run dream generation!
) else (
    echo ❌ OpenAI API key is NOT set!
    echo.
    echo To set it for this session:
    echo   set OPENAI_API_KEY=your-api-key-here
    echo.
    echo To set it permanently:
    echo   setx OPENAI_API_KEY "your-api-key-here"
    echo.
    echo ⚠️  After using setx, restart your terminal/PowerShell
)

echo.
pause 