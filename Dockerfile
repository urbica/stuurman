FROM python:2.7
MAINTAINER Stepan Kuzmin <to.stepan.kuzmin@gmail.com>

RUN apt-get -yqq update \
  && apt-get -yqq install \
  libgeos-dev \
  libspatialindex-dev

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

EXPOSE 5000
CMD ["python", "server.py"]
