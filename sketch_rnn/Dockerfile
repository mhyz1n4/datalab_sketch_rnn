FROM python:2

RUN pip install scipy matplotlib intervaltree bokeh magenta requests

RUN mkdir -p /sketch_rnn_floder
WORKDIR /sketch_rnn_floder

RUN git clone https://github.com/mhyz1n4/datalab_sketch_rnn.git

WORKDIR /sketch_rnn_floder/datalab_sketch_rnn/sketch_rnn

RUN touch checkpoints

# expose port
EXPOSE 8080
