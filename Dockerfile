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

# Copiar requirements e instalar dependencias usando el índice CPU ligero de PyTorch
COPY detect-altur/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copiar la aplicación
COPY detect-altur /app/detect-altur

WORKDIR /app/detect-altur

# Puerto dinámico (Render asigna $PORT automáticamente)
ENV PORT=8000
EXPOSE 8000

# Comando para iniciar Uvicorn evaluando el puerto dinámico $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
