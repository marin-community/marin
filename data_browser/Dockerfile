FROM python:3.12

# Where to install stuff
WORKDIR /app

# Install poetry
RUN pip install poetry
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies
COPY pyproject.toml poetry.lock .
RUN poetry install --no-root

# Copy app
COPY server.py server.py
COPY build build

EXPOSE 5000

CMD ["poetry", "run", "python", "server.py"]
