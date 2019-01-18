FROM ubuntu:18.04

# Bring system up-to-date
RUN apt-get update && apt-get upgrade -y


# Install required packages and remove the apt packages cache when done.
RUN apt-get install -y --no-install-recommends \
    build-essential \
	libcv-dev \
	libjpeg-dev \
	libjpeg8-dev \
	libleptonica-dev \
	libopencv-dev \
	libpng12-dev \
	libtesseract-dev \
	libtiff5-dev \
	libtool \
	pkg-config \
	python3 \
	python3-dev \
	python3-setuptools \
	python3-pip \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-ita \
    tesseract-ocr-spa \
	zlib1g-dev


# Copy scripts into the container for convenience (you could mount a folder as well)
RUN mkdir -p /pipeline/{tasks,uploads,failed}

WORKDIR /pipeline

COPY . /pipeline

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED 1

# EXPOSE 8081

# Clear out the apt cache
RUN rm -rf /var/lib/apt/lists/*
