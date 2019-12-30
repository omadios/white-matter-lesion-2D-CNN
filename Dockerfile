FROM python:3.6
WORKDIR /app
COPY . /
RUN apt-get update
RUN pip install -r requirements.txt
CMD [ "bash", "src/antani/server.sh" ]
