#FROM python:3.10.8
#COPY . /app
#WORKDIR /app
#RUN pip install -r requirements.txt
#CMD python app.py

FROM python:3.10.8

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .  

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

