#! make

include .env


GIT_LAST_VERS = $(shell git tag -l v* | tail -n1)


BUILD_PIPELINE 	= pipeline
BUILD_WEBAPP 	= webapp

CONTAINER_PIPELINE = $(BUILD_PIPELINE)-instance
CONTAINER_WEBAPP = $(BUILD_WEBAPP)-instance

CMD_RUN_PIPELINE = python3 /app/app.py --run pipeline
CMD_RUN_WEBAPP = python /app/app.py --run web




build-all: build-pipeline build-webapp
build-pipeline: 
	docker-compose up --build --no-deps $(BUILD_PIPELINE)

build-webapp: 
	docker-compose up --build --no-deps $(BUILD_WEBAPP)


shell-pipeline:
	docker-compose run -d $(BUILD_PIPELINE) /bin/bash


run-all: run-pipeline run-webapp
run-pipeline:
	docker-compose run -d --name $(CONTAINER_PIPELINE) --entrypoint $(BUILD_PIPELINE) $(CMD_RUN_PIPELINE)
run-webapp:
	docker-compose run -d --name $(CONTAINER_WEBAPP) --entrypoint $(BUILD_WEBAPP) $(CMD_RUN_WEBAPP)


stop-all : stop-pipeline stop-webapp
stop-pipeline:
	docker-compose stop $(BUILD_PIPELINE)
stop-webapp:
	docker-compose stop $(BUILD_WEBAPP)


rm-all: rm-pipeline rm-webapp
rm-pipeline:
	docker rm $(CONTAINER_PIPELINE)
rm-webapp:
	docker rm $(CONTAINER_WEBAPP)


all: test 

# .PHONY: build run 
