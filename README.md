# HackSabamApp

App for Sabam Use Case on IBM Hackathon 2018

## Development Setup

You need to install Docker, and docker-compose.

Copy the _.env.example_ file to _.env_, and set the config for your use.
That file is not tracked by git.

Bringing up a development environment from the repository consists of the following commands:

```
$ docker-compose build
$ docker-compose up
```

Access it on [localhost:5000](localhost:5000)

### Interact Directly with thle SQL DB

```
$ docker-compose exec db psql --user ${DB_SQL_USER} ${DB_SQL_PWD}
```

### Add New Dependencies

```
$ docker-compose exec app bash
...
$ pip freeze --local > requirements.txt
```

### The App Container Crashes - How Do I Find Out What's Wrong?

It tries to start the dev server right away for convenience.
If that crashes for some reason, the container stops as well.
Here's what you need to be able to debug it in such a case.

Bring the stack down and up again. Now the container is running,
and you can exec into it and fix the issue, so it works again.

```
$ docker-compose exec app bash
```
