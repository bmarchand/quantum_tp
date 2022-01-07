FROM ubuntu:latest

RUN apt-get update && apt -y update
RUN apt-get -y install python3.9-dev
RUN apt-get -y install python3-pip
RUN python3.9 -m pip install notebook
RUN python3.9 -m pip install myqlm
RUN apt -y install imagemagick
RUN python3.9 -m qat.magics.install

RUN mkdir src
WORKDIR src/
COPY . .

CMD ["cat","/etc/fstab"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
