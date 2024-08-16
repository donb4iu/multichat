# Use Python 3.12.3 as the base image
FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "chat_pdf.py"]
