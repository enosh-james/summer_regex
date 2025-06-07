# Streamlit Application 
# Requirements.txt(streamlit , google-generativeai , python-dotenv)
# .env file (API_KEY)
# App.py file 

# localhost ---> Run 8501 Port 

# step-1 
# Create a github repo of your application 

# https://github.com/1602saurab/summer-bot.git

# step-2 
# Go to AWS ---> 
# 2 EC2 SERVER (1. CI SERVER   2. DEPLOYMENT SERVER)

# 2.1. CI SERVER ----> Install docker  , clone github repository , create docker image , docer login, push image to docker hub . 



# IN CI-SERVER 
# sudo apt-get update 
# sudo apt-get install docker.io -y 
# sudo usermod -aG docker $USER && newgrp docker 

# vim Dockerfile 

# press ---> i 
# -------------------------------------------------------------
# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy requirements.txt and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code
# COPY . .

# # Expose the Streamlit port
# EXPOSE 8501

# # Run the Streamlit application
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# -------------------------------------------------------------------
# press ---> esc ---> :wq 

# sudo systemctl status docker

# docker build -t your_dockerhub_name/bot_name . 
# docker build -t your_dockerhub_nam/qery . 

# docker login 

# docker push your_dockerhub_nam/bot_name:latest 
