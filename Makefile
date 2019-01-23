#! make

include .env


GIT_LAST_VERS = $(shell git tag -l v* | tail -n1)


BUILD_PIPELINE 	= pipeline
BUILD_WEBAPP 	= webapp



build: build-pipeline build-webapp

build-pipeline: 
	docker-compose up $(BUILD_PIPELINE)

build-webapp: 
	docker-compose up $(BUILD_WEBAPP)


shell-pipeline:
	docker-compose run --rm $(BUILD_PIPELINE) /bin/bash

shell-webapp:
	docker-compose run --rm $(BUILD_WEBAPP) /bin/bash


run: run-pipeline run-webapp

run-pipeline:
	docker-compose run -d --name $(BUILD_PIPELINE) python3 app.py --run pipeline

run-webapp:
	docker-compose run -d --name $(BUILD_WEBAPP) python3 app.py --run web


stop : stop-pipeline stop-webapp

stop-pipeline:
	docker-compose stop $(BUILD_PIPELINE)

stop-webapp:
	docker-compose stop $(BUILD_WEBAPP)


rm: rm-pipeline rm-webapp

rm-pipeline:
	docker rm rpi-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)

rm-webapp:
	docker rm rpi-$(CONTAINER_NAME)-$(CONTAINER_INSTANCE)


all: test 

.PHONY: build run 
