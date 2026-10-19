FROM mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates coreutils curl findutils git procps \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/agentenv-stand-in/
RUN pip install --no-cache-dir -r /opt/agentenv-stand-in/requirements.txt && useradd --create-home agent
COPY agentenv_protocol/ /opt/agentenv-stand-in/agentenv_protocol/
ENV PYTHONPATH=/opt/agentenv-stand-in
WORKDIR /app
COPY a2a_server.py .
RUN chown agent:agent /app
USER agent
CMD ["python", "/app/a2a_server.py"]
