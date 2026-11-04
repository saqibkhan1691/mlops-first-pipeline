# pull the python image from docker hub
# use a lightweight python image
FROM python:3.14-slim

# set the working directory inside container
WORKDIR /app

# copy dependencies list before copying application code
COPY requirements.txt .

# install the dependencies
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and saved model insdie /app
COPY . .

# document the port used by the FastAPI application
EXPOSE 8000

# start the API server when container is ready
# 0.0.0.0 - allows connections through the container's network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]