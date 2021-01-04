FROM python:3.8-alpine

WORKDIR /notebooks

COPY minimal_notebook.ipynb ./notebooks

RUN apk update
RUN apk add build-base
RUN apk add libffi-dev
RUN pip install cython
RUN apk add zeromq
RUN apk add zeromq-dev
RUN pip install pyzmq
RUN pip install notebook
RUN pip install myqlm

CMD ["jupyter-notebook","--no-browser","--port","8080"]
