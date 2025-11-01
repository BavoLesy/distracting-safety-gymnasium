
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
#nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# Set non-interactive mode to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Copy the requirements file to the working directory
RUN mkdir /opt/python3.8

# To avoid .pyc files and save space
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install all dependencies you need to compile Python3.8
RUN apt update
RUN apt install -y wget libffi-dev gcc build-essential curl tcl-dev tk-dev uuid-dev libbz2-dev lzma-dev liblzma-dev libssl-dev libsqlite3-dev wget libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf xvfb

#install git
RUN apt install git -y 

ENV MUJOCO_GL=egl
# Download Python source code from official site and build it
RUN wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
RUN tar -zxvf Python-3.8.10.tgz
RUN cd Python-3.8.10 && \
    ./configure --prefix=/opt/python3.8 && \
    make && make install && \
    /opt/python3.8/bin/python3.8 -m ensurepip

# Delete the python source code and temp files
RUN rm Python-3.8.10.tgz
RUN rm -r Python-3.8.10/

# Now link it so that $python works
RUN ln -s /opt/python3.8/bin/python3.8 /usr/bin/python
RUN ln -s /opt/python3.8/bin/pip3 /usr/bin/pip
# add to path
ENV PATH /opt/python3.8/bin:$PATH
#install mujoco
RUN mkdir -p /root/.mujoco && \

    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

# Set environment variables for MuJoCo
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:$LD_LIBRARY_PATH
ENV MUJOCO_LICENSE_KEY_PATH=/root/.mujoco/mjkey.txt
#download dataset for distractions video background 
RUN wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
RUN apt-get update && apt-get install -y unzip \
    && unzip DAVIS-2017-trainval-480p.zip && rm DAVIS-2017-trainval-480p.zip

COPY requirements.txt .
COPY install.sh .
COPY . .

RUN pip install --upgrade pip && \
    FAILED=/tmp/pip_failed.txt; : > "$FAILED"; \
    while IFS= read -r line; do \
        case "$line" in \
            \#*|'' ) continue ;; \
        esac; \
        if ! pip install --default-timeout=100 "$line"; then \
            echo "$line" >> "$FAILED"; \
        fi; \
    done < requirements.txt; \
    if [ -s "$FAILED" ]; then \
        echo "The following requirements failed to install:"; \
        cat "$FAILED"; \
    else \
        echo "All requirements installed successfully."; \
    fi; \
    true