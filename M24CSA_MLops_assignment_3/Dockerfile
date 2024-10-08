FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /workspace

# Copy all the project files into the container
COPY . .

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 
EXPOSE 80

# Environment variable (optional)
ENV NAME MLOpsLab

# Command to run the training script when the container starts
CMD ["python", "train.py"]

