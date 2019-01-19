FROM ubuntu:18.04

# Ensure that we always use UTF-8 and with English GB locale
# RUN locale-gen en_GB.UTF-8

ENV TZ=Europe/Brussels
ENV PYTHONIOENCODING=utf-8
# ENV LC_ALL=en_GB.UTF-8
# ENV LANG=en_GB.UTF-8
# ENV LANGUAGE=en_GB.UTF-8


# Bring system up-to-date
RUN apt-get update && apt-get upgrade -y


# Install required packages and remove the apt packages cache when done.
RUN apt-get install -y --no-install-recommends \
	build-essential apt-utils\
	# libjpeg-dev \
	# libjpeg8-dev \
	# libleptonica-dev \
	# libopencv-dev \
	# libtesseract-dev \
	# libtiff5-dev \
	# libtool \
	# pkg-config \
	python3 \
	python3-dev \
	python3-setuptools \
	python3-pip \
	tesseract-ocr \
	tesseract-ocr-eng \
	tesseract-ocr-fra \
	tesseract-ocr-deu \
	tesseract-ocr-ita \
	tesseract-ocr-spa 
# zlib1g-dev


# Copy scripts into the container for convenience (you could mount a folder as well)
RUN mkdir -p /pipeline/{tasks,uploads,failed}

WORKDIR /pipeline

COPY . /pipeline

COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED 1

# EXPOSE 8081

# Clear out the apt cache
RUN apt-get -y clean autoclean \
	&& apt-get -y autoremove \
	&& rm -rf /var/lib/{apt,dpkg,cache,log}/
