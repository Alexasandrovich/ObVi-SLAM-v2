#!/bin/bash
xhost +local:docker
docker compose -f docker/docker-compose.yml up --build --force-recreate