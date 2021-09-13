FROM python:3.7

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y

COPY src/ src/
WORKDIR src/

RUN pip install -r requirements.txt

ENV PORT 8080

CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 app:app

