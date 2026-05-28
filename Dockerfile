FROM python:3.10-slim

RUN pip install qpsolvers[open_source_solvers]==4.8.2 numpy==1.26.4

WORKDIR /app

COPY libs libs

RUN apt update && apt install -y build-essential

RUN cd libs && cd robotics-toolbox-python && pip install -e .
RUN cd libs && cd swift && pip install -e .

RUN pip install tqdm matplotlib

RUN pip install qpsolvers[proxqp]

RUN useradd -m -s /bin/bash taco

COPY --chown=taco: . .
