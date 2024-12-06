import subprocess
import os

# Define the path to app.py
app_file = os.path.join('app', 'app.py')

# Run the Streamlit app
subprocess.run(['streamlit', 'run', app_file])
