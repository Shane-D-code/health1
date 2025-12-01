@echo off
echo Starting HealthCare Analysis Application...

echo Installing dependencies...
cd FrontEnd
call npm install
cd ..

cd health-backend-java\health-backend
pip install -r requirements_integrated.txt
cd ..\..

echo Starting backend server...
start "Backend" cmd /k "cd health-backend-java\health-backend\app && python integrated_main.py"

timeout /t 3

echo Starting frontend...
start "Frontend" cmd /k "cd FrontEnd && npm run dev"

echo Application started! 
echo Backend: http://localhost:8001
echo Frontend: http://localhost:5173
pause