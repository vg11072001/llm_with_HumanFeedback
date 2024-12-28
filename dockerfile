# Use the NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set the timezone to Asia/Tokyo (optional, adjust if needed)
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Update and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libreadline-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libbz2-dev \
    liblzma-dev \
    zlib1g-dev \
    uuid-dev \
    libffi-dev \
    libdb-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    ffmpeg \
    unzip \
    && apt-get clean

# Install Python 3.10 (if not already available in the base image)
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz \
    && tar -xf Python-3.10.13.tgz \
    && cd Python-3.10.13 \
    && ./configure --enable-optimizations \
    && make \
    && make install

# Set the working directory
WORKDIR /app

# Set the Python environment path (optional)
ENV PYTHONPATH=/app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install the Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Copy the run_all.sh script into the container
COPY run_all.sh .

# Make the script executable
RUN chmod +x run_all.sh

# Command to run the container (optional)
CMD ["python3"]
