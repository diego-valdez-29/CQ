# Imagen base de Python 3.11 en versión slim
FROM python:3.11-slim

# Evitar la creación de archivos .pyc y asegurar logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema requeridas por soundfile y librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY detect-altur/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la aplicación
COPY detect-altur /app/detect-altur

WORKDIR /app/detect-altur

# Exponer el puerto de la API (8000 por defecto)
EXPOSE 8000

# Comando para iniciar la aplicación con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
