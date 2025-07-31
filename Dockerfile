FROM ubuntu:24.04 AS build

LABEL maintainer="zazamaza"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget tar gcc ca-certificates libc6-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

ENV GO_RELEASE=1.23.1

RUN wget https://dl.google.com/go/go${GO_RELEASE}.linux-amd64.tar.gz \
    && tar xfv go${GO_RELEASE}.linux-amd64.tar.gz -C /usr/local \
    && rm go${GO_RELEASE}.linux-amd64.tar.gz

ENV PATH="${PATH}:/usr/local/go/bin"

WORKDIR /app
