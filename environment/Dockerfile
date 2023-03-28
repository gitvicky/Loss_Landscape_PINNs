# hash:sha256:96adb2f5dbee72742f3c6e1a6ae1764c79be1e16f05e299c246b5fd84dd45201
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    pandas==1.5.3 \
    pydoe==0.3.8 \
    torch==1.8.0 \
    tqdm==4.65.0

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
