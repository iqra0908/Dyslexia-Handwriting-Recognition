
#!/usr/bin/env bash

# Install libgl1-mesa-glx package
apt-get update
apt-get install -y libgl1-mesa-glx

# Start the app
gunicorn app:app --bind=0.0.0.0:$PORT --workers=1 --threads=8 --timeout=0
