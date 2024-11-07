# python image
FROM python:3.10-slim

# workdir
WORKDIR /app

# copy files
COPY . /app

# dependencies
RUN pip install --no-cache-dir -r requirements.txt

# model name
ENV MODEL_NAME=annyalvarez/Roberta-MoralPres-MS-DM-P2

# Expone el puerto 8000 para que el servidor esté accesible
EXPOSE 8000

# run server
CMD ["python", "server.py"]