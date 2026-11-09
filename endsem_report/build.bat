@echo off
setlocal
cd /d "%~dp0"
echo Running first pdflatex pass...
pdflatex -interaction=nonstopmode Major_Project_Endsem_Report.tex
if errorlevel 1 goto :error
echo Running second pdflatex pass...
pdflatex -interaction=nonstopmode Major_Project_Endsem_Report.tex
if errorlevel 1 goto :error
echo.
echo Build complete: Major_Project_Endsem_Report.pdf
exit /b 0
:error
echo.
echo Build failed. See Major_Project_Endsem_Report.log for details.
exit /b 1
