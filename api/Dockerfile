FROM ubuntu:latest
LABEL authors="nuc"

ENTRYPOINT ["top", "-b"]FROM python:3.11-slim

WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la API
COPY . .

# Copiar el modelo y scaler desde la carpeta notebook
COPY ../notebook/models/modelo_final.pkl ./models/modelo_final.pkl
COPY ../notebook/models/scaler.pkl ./models/scaler.pkl

# Exponer puerto
EXPOSE 8000

# Comando para arrancar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]