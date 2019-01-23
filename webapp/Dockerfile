FROM python:3.6.8-alpine

# bring system up-to-date
RUN apk update

RUN apk add --no-cache --virtual .build-deps \
    build-base \
    && find /usr/local \
    \( -type d -a -name test -o -name tests \) \
    -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
    -exec rm -rf '{}' + \
    && runDeps="$( \
    scanelf --needed --nobanner --recursive /usr/local \
    | awk '{ gsub(/,/, "\nso:", $2); print "so:" $2 }' \
    | sort -u \
    | xargs -r apk info --installed \
    | sort -u \
    )" \
    && apk add --virtual .rundeps $runDeps \
    && apk del .build-deps

# copy scripts into the container for convenience (you could mount a folder as well)
RUN mkdir /app/

COPY .env config.py app.py webapp/requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONUNBUFFERED 1

VOLUME [ "/app/webapp", "/app/data/tasks", "/app/data/uploads" ]

EXPOSE ${WEBAPP_PORT}

# Clear out the apk cache
RUN apk del build-base && rm -rf /var/cache/apk/*

