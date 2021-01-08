FROM ubuntu:latest

WORKDIR /notebooks

COPY minimal_notebook.ipynb ./notebooks

RUN apt-get update
RUN apt-get -y install python3-dev python3-pip
RUN pip3 install notebook
RUN pip3 install myqlm
RUN python3 -m qat.magics.install

CMD ["jupyter-notebook","--no-browser","--port","8080"]
