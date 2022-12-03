FROM ghcr.io/pytorch/pytorch-nightly
RUN apt update && apt install -y git
WORKDIR /workspace
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt
ENTRYPOINT [ "/bin/bash" ]