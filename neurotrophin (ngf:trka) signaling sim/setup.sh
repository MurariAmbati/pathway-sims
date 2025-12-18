#!/bin/bash
# installation and setup script for ngf/trka signaling simulator

echo "setting up neurotrophin (ngf/trka) signaling simulator..."

# create virtual environment
echo "creating virtual environment..."
python3 -m venv venv

# activate virtual environment
echo "activating virtual environment..."
source venv/bin/activate

# upgrade pip
echo "upgrading pip..."
pip install --upgrade pip

# install requirements
echo "installing dependencies..."
pip install -r requirements.txt

echo "setup complete!"
echo ""
echo "to run the simulator:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
