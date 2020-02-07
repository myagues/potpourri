FROM tensorflow/tensorflow:latest-gpu-py3

ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USERNAME=tf2

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y install --no-install-recommends git \
    # Create a non-root user to use if preferred
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /go/src /tmp/gotools
ENV DEBIAN_FRONTEND=dialog

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir jupyterlab \
    ipywidgets \
    jupyterlab \
    jupyter_http_over_ws \
    && export VERSION=v12.14.1 DISTRO=linux-x64 && mkdir -p /nodejs \
    && curl -SL https://nodejs.org/dist/$VERSION/node-$VERSION-$DISTRO.tar.xz | tar -xJC /nodejs \
    && export PATH=/nodejs/node-$VERSION-$DISTRO/bin:$PATH \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter serverextension enable --py jupyter_http_over_ws \
    && python -m ipykernel.kernelspec

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

CMD ["bash", "-c", "jupyter lab --ip 0.0.0.0 --no-browser"]
