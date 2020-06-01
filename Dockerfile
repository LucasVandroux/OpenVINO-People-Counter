FROM ubuntu:18.04
LABEL Name=openvino-app Version=0.0.1
USER root
WORKDIR /
RUN useradd -ms /bin/bash openvino && \
    chown openvino -R /home/openvino
ARG DEPENDENCIES="autoconf \
                  automake \
                  build-essential \
                  cmake \
                  cpio \
                  curl \
                  gnupg2 \
                  libdrm2 \
                  libglib2.0-0 \
                  lsb-release \
                  libgtk-3-0 \
                  libtool \
                  python3-pip \
                  udev \
                  unzip"
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*
ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz
ARG INSTALL_DIR=/opt/intel/openvino
WORKDIR /tmp
RUN curl -LOJ "${DOWNLOAD_LINK}" && \
    tar -xzf ./*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf /tmp/*
RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh