FROM python:3.9.5

#application port
EXPOSE 5010

# set noninteractive mode to fix errors related to apt-get
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN echo "Africa/Cairo" > /etc/timezone

#Download pacjages
RUN apt update -- && apt install -y \
    python \
    git \
    python3-pip \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender1

# Install pip requirements
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /root/

WORKDIR /root/

# ENTRYPOINT  python3 inference.py

ENV FLASK_APP="inference.py" FLASK_DEBUG=1 FLASK_ENV=docker
CMD ["python3", "-m" ,"flask", "run", "-h", "0.0.0.0"]