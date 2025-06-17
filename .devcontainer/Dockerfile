
FROM osrf/ros:humble-desktop AS base


ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    build-essential \
    cmake \
    g++ \
    curl \
    wget \
    lsb-release \
    gnupg2 \
    openssh-client \
    libusb-1.0-0-dev \
    ros-humble-tf-transformations \
    ros-humble-rosbridge-server \
    gedit \
    iputils-ping \
    net-tools \
    tree \
    usbutils \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ur \
    iptables \
    && rm -rf /var/lib/apt/lists/*


RUN test -f /etc/ros/rosdep/sources.list.d/20-default.list || rosdep init && rosdep update

RUN curl -L micro.mamba.pm/install.sh | bash


RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /root


CMD ["/bin/bash"]
