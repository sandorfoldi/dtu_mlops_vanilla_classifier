# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git

# COPY requirements.txt requirements.txt
# COPY setup.py setup.py
# COPY src/ src/
# COPY data/ data/
# COPY models/ models/
# COPY reports/ reports
COPY ./ /

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip list

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

