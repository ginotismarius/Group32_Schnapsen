REM Create a python virtual install in this directory
REM and install required packages from requirements.txt
python -m venv venv
call venv/Scripts/activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt




