
ARG PYTHON_VERSION=3.11.11
FROM python:${PYTHON_VERSION}-slim as base

WORKDIR /API

COPY ./requirements.txt /API
RUN  pip install -r requirements.txt

# Copy the source code into the container.
COPY . /API


# Run the application.
CMD python main.py