@echo off
echo Heart Disease App - Debug Mode
echo ================================
echo.

REM Activate virtual environment
call heart_disease_env\Scripts\activate

echo Checking dependencies...
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)"
python -c "import reportlab; print('✅ ReportLab:', reportlab.Version)"
python -c "from utils.report_generator import ReportGenerator; print('✅ ReportGenerator: OK')"
echo.

echo Starting Streamlit app in debug mode...
echo The app will open at: http://localhost:8501
echo.
echo DEBUG FEATURES ENABLED:
echo - Detailed PDF generation logging
echo - Multiple download methods
echo - Debug information panel
echo - Alternative download server
echo.

streamlit run app.py --server.headless false --server.port 8501 --logger.level debug

pause
