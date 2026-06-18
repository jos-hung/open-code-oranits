#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo " Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo " No requirements.txt found. Skipping dependency installation."
fi

if [ ! -d "task" ]; then
    echo "Creating 'task' directory..."
    mkdir task
else
    echo "'task' directory already exists."
fi

echo "Running ITS_based.py..."
python ./src/physic_definition/system_base/ITS_based.py
