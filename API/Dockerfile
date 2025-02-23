# Use slim Python base image
ARG PYTHON_VERSION=3.10.11
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /api

# Use root user to install system dependencies
USER root

# Ensure the package lists are clean and update system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY ./requirements.txt /api
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-privileged user after installing dependencies
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Switch to the non-privileged user to run the application
USER appuser 

# Copy the source code into the container
COPY . /api

# Expose the port that the application listens on
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
