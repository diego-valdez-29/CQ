# detect-altur API + Docker + Cloudflare Tunnel

API REST basada en **FastAPI** (`detect-altur`) para la detección de voz sintética vs. humana en llamadas telefónicas. Incluye soporte para **Docker Compose** y exposición pública mediante un túnel de Cloudflare (`cloudflared`).

---

## 📁 Estructura del Proyecto

```text
.
├── detect-altur/        # Código fuente de la API (FastAPI) y detectores
├── audios/              # Muestras de audio para pruebas
├── Dockerfile           # Construcción de la imagen Docker (Python 3.11 + libsndfile1 + ffmpeg)
├── docker-compose.yml   # Orquestación de API FastAPI + Túnel Cloudflare
├── .env.example         # Plantilla de variables de entorno
└── .gitignore           # Archivos ignorados por Git
```

---

## 🚀 Cómo Iniciar la Aplicación

### 1. Iniciar con Docker Compose (Modo Quick Tunnel)
Por defecto, el túnel generará una URL pública temporal de tipo `https://xxxx.trycloudflare.com` sin requerir credenciales adicionales.

```bash
docker compose up -d --build
```

### 2. Ver la URL Pública del Túnel de Cloudflare
Para obtener el enlace público generado por Cloudflare:

```bash
docker compose logs -f tunnel
```

Verás una línea en los logs similar a:
```text
+-----------------------------------------------------------------------------------+
| Your quick Tunnel has been created! Visit it at:                                  |
| https://random-subdomain.trycloudflare.com                                        |
+-----------------------------------------------------------------------------------+
```

### 3. Probar la API

- **Local:** `http://localhost:8000/detect`
- **Túnel Público:** `https://<tu-subdominio>.trycloudflare.com/detect`

#### Ejemplo de llamada `POST /detect` (Base64)
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$(base64 -w0 audios/prueba.wav)\"}"
```

#### Respuesta de la API:
```json
{
  "is_synthetic": true,
  "confidence": 0.8321
}
```

---

## 🔑 Configurar Túnel con Dominio Propio (Cloudflare Zero Trust Token)

Si cuentas con un token de túnel persistente en Cloudflare Zero Trust:

1. Crea tu archivo `.env`:
   ```bash
   cp .env.example .env
   ```
2. Añade tu token a `.env`:
   ```text
   TUNNEL_TOKEN=tu_token_aqui
   ```
3. En `docker-compose.yml`, ajusta la sección del servicio `tunnel` desmarcando la opción de `tunnel run` y `TUNNEL_TOKEN`.
