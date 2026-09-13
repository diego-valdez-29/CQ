# detect-altur API + Docker + Cloudflare Tunnel + Render

API REST basada en **FastAPI** (`detect-altur`) para la detección de voz sintética vs. humana en llamadas telefónicas.

Soporta despliegue en **Render.com**, **Docker Compose** y exposición mediante **Túnel de Cloudflare** (`cloudflared`).

---

## 🌐 Cómo Desplegar en Render (render.com)

### Opción A: Despliegue Automático con Blueprint (Recomendado)
1. Entra a tu panel de **Render** (`dashboard.render.com`).
2. Haz clic en **New +** -> **Blueprint**.
3. Conecta tu repositorio de GitHub `diego-valdez-29/comoquieras`.
4. Render detectará automáticamente el archivo `render.yaml` y creará el Web Service con Docker.
5. Haz clic en **Apply**. ¡Y listo! Render desplegará tu API y te dará una URL pública tipo `https://detect-altur-api.onrender.com`.

### Opción B: Despliegue Manual como Web Service
1. En Render Dashboard, selecciona **New +** -> **Web Service**.
2. Conecta tu repositorio de GitHub.
3. Elige la opción **Docker** en el Runtime (usará el `Dockerfile` raíz).
4. Elige el plano **Free** o Starter.
5. Haz clic en **Create Web Service**.

---

## 🚀 Despliegue Local con Docker Compose

```bash
docker compose up -d --build
```

- **API Local**: `http://localhost:8000/detect`
- **Túnel Cloudflare Persistente**: `https://api-como-quieras.devs-dom.com/detect`

---

## 🧪 Ejemplo de llamada `POST /detect`

```bash
curl -X POST https://tu-servicio.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$(base64 -w0 audios/prueba.wav)\"}"
```

#### Respuesta de la API:
```json
{
  "is_synthetic": false,
  "confidence": 0.5
}
```
>>>>>>> claude/feature/evento
