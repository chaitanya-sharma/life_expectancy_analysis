FROM python:3.11-slim-bookworm
COPY . /app
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=3600  
RUN pip install -r requirements.txt
CMD python app.py