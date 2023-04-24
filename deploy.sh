
#!/usr/bin/env bash

# Give execute permission to the script file
chmod +x /home/site/wwwroot/deploy.sh

# Install required packages
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install -r requirements.txt
apt-get install -y libgl1-mesa-glx

# Start the application
gunicorn app:app --bind=0.0.0.0:$PORT --workers=4
