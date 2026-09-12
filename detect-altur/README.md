# detect-altur

Detecta si el *caller* de una llamada telefónica es una persona real o una voz sintética (agente de IA haciéndose pasar por humano).

## Contrato del endpoint

`POST /detect` — ruta que califica el reto.

**No se pudo confirmar con los organizadores el schema exacto del body**, así que el endpoint es deliberadamente permisivo con el formato de entrada y acepta varias formas de mandar el WAV en base64:

1. **JSON con alguno de estos nombres de campo** (case-insensitive — no importa si llega `audio_base64`, `Audio_Base64` o `AUDIOBASE64`; se usa el primero que encuentre, en este orden):

   ```
   audio_base64, audio, wav_base64, wav, data, file, audioBase64, audio_data, base64_audio, file_base64
   ```

2. **Body crudo como string base64 directo**, sin envolver en JSON: si el `Content-Type` de la request no es `application/json`, o si el body no parsea como JSON válido, el endpoint toma el body completo tal cual y lo intenta decodificar directamente como base64 (por si Altur manda el WAV como texto plano).

Si ninguna de las dos formas produce un base64 válido, responde `422`:
- con el detalle de qué campos acepta y qué keys llegaron, si se encontró un JSON pero ningún campo coincidió;
- con `"base64 invalido: ..."` si se encontró un candidato (de un campo o del body crudo) pero no decodifica como base64 válido.

El WAV debe venir en 2 canales: **canal 0 = caller** (la persona/voz a clasificar) y **canal 1 = agente** (el bot de Altur, conocido, no se evalúa). Si el WAV viene mono, ambos canales se toman como el mismo audio.

Respuesta:

```json
{"is_synthetic": true, "confidence": 0.83}
```

`POST /detect-dev` — variante solo para desarrollo local: recibe el audio como `multipart/form-data` (`file=@audio.wav`) en vez de base64. No es la ruta que evalúan los jueces.

## Las tres señales

- **Acústico** (`app/deteccion/acustico.py`): heurística sobre MFCCs del canal del caller (varianza y regularidad espectral), normalizada por z-score y pasada por sigmoide. Es un *fallback*: no hay certeza del checkpoint exacto de un modelo tipo AASIST, así que se usa esta heurística en su lugar.
- **Comportamiento** (`app/deteccion/comportamiento.py`): mide qué tan *consistente* es el tiempo que tarda el caller en retomar el habla después de que el agente provoca un silencio largo o un atropello. Recuperación muy uniforme (std baja) se interpreta como sospechosa de máquina; un humano se recupera de forma más errática.
- **Semántico** (`app/deteccion/semantico.py`): transcribe el canal del caller con Whisper y busca honestidad ante preguntas trampa (información que no existe). Si el transcript ya dice algo tipo "no sé"/"no tengo esa información" (regex), se resuelve como humano sin llamar al LLM. Si no, se manda a un LLM (la Spark, vía Tailscale) a clasificar si la respuesta suena a dato inventado (confabulación → sintético).

**Fusión**: promedio ponderado de las tres señales (`app/deteccion/fusion.py`). Los pesos en producción son un *placeholder* `{acustico: 1.0, comportamiento: 1.0, semantico: 1.0}` con umbral `0.5` — **no** los pesos que salen de la calibración (ver siguiente sección).

**Decisión secuencial**: en vez de esperar la llamada completa, se evalúa en checkpoints de **25s** y **40s** de audio; si la confidence en un checkpoint cruza `UMBRAL_DECISION_TEMPRANA = 0.75`, se responde ahí sin esperar más. Si ningún checkpoint cruza el umbral, decide al final de la llamada.

## Estado honesto de calibración

Se validó contra una muestra balanceada de **n=30 llamadas reales** (15 humanas / 15 sintéticas) del split `train` del dataset del reto (`manifest.csv`, 353 llamadas reales en total; `train` tiene 282). Resultados en [`calibracion/calibracion_resultados_v2.jsonl`](calibracion/calibracion_resultados_v2.jsonl), reproducibles corriendo el propio `calibrar.py` sobre ese archivo:

| Señal | media humano | media sintético | \|diff\| |
|---|---|---|---|
| acústico | 0.526 | 0.446 | 0.080 |
| comportamiento | 0.044 | 0.012 | 0.032 |
| semántico | 0.453 | 0.477 | 0.023 |

Ninguna de las tres separa bien en esta muestra, y las dos primeras separan **al revés** de la convención (score más alto = más sintético): la regresión logística de calibración les asigna coeficiente negativo, así que `calibrar_pesos()` les pone peso 0 y deja todo el peso en semántico (que a su vez es poco informativa en esta corrida — ver abajo). Con los pesos calibrados así (`acustico=0, comportamiento=0, semantico=1.0`, umbral 0.15): accuracy 53.3%, EER 46.7% en la propia muestra de calibración. Con los pesos placeholder que sí están en producción (1/1/1, umbral 0.5): accuracy 46.7%, EER 56.7% — peor que adivinar al azar.

Por señal, específicamente:

- **Semántico** estuvo casi todo el tiempo en su valor neutro de fallback (`score=0.5`, `llm_no_disponible`) en **27 de las 30 llamadas** — la corrida de calibración no tuvo acceso confiable a la Spark. Solo 3 llamadas resolvieron por el filtro regex (`score=0.15`). Es decir: la señal semántica está prácticamente sin validar con el LLM real; lo único probado es su ruta de fallback.
- **Comportamiento** saturó en `score=0.0` en 12/15 llamadas humanas y 14/15 sintéticas — casi sin separación, señal débil documentada también en el propio código (`ESCALA_NORMALIZACION_STD` sin calibrar contra dataset real).
- **Acústico** es la única señal con algo de rango, pero con separación débil y en la dirección incorrecta, documentado en el código como señal secundaria no decisiva por sí sola (`separacion_debil: True` en el detalle de cada respuesta).
- Con los pesos calibrados, **0 de las 30 llamadas** habrían cruzado el umbral de decisión temprana (`0.75`) en los checkpoints de 25s/40s — la decisión temprana nunca se disparó en esta muestra.

En resumen: el pipeline corre de punta a punta y el contrato del endpoint está probado, pero la calidad de la clasificación en sí **no está validada** contra el dataset real con los pesos que corren en producción — ver Limitaciones.

## Cómo correrlo

```bash
cd detect-altur
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # y llenar SPARK_API_KEY

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Variables de entorno (ver `.env.example`):

- `SPARK_API_KEY` — token para el LLM de la señal semántica (`http://100.122.49.1:8080/api`, alcanzable por Tailscale). Si no está seteada o la Spark no responde, la señal semántica cae a su fallback neutro (`score=0.5`), no truena el request.

Ejemplo con `/detect` (base64):

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$(base64 -w0 audios/prueba.wav)\"}"
```

Ejemplo con `/detect-dev` (multipart, más cómodo para pruebas locales):

```bash
curl -X POST http://localhost:8000/detect-dev -F "file=@audios/prueba.wav"
```

## Exposición para evaluación

Pendiente de definir cómo se expone la URL pública para que los jueces la llamen (candidato natural: Tailscale Funnel, ya que la infraestructura de la Spark corre sobre la misma red Tailscale). *(sección a completar cuando se decida el método final)*

## Limitaciones conocidas

- **Dependencia de la Spark**: la señal semántica depende de un LLM externo alcanzable solo por Tailscale, con timeout de 2s. Si no está disponible cae a `score=0.5` (neutro) — el pipeline no falla, pero pierde la señal más prometedora conceptualmente. En la calibración esto pasó en 27/30 llamadas.
- **Señal acústica débil en 8kHz**: validado con datos reales (n=30) — la heurística MFCC no separa bien humano/sintético en telefonía de 8kHz, y en esta muestra separa en la dirección contraria a la esperada. No debe tratarse como decisiva.
- **Señal de comportamiento sin calibrar**: `ESCALA_NORMALIZACION_STD` es un valor de partida sin ajustar contra el dataset real; en la validación satura en 0.0 para la gran mayoría de llamadas de ambas clases.
- **Pesos de fusión en producción no son los calibrados**: `app/main.py` sigue usando el placeholder `1/1/1` con umbral `0.5`, no los pesos que salen de `calibrar_pesos()`. Aplicar los pesos calibrados de esta muestra (que ponen todo el peso en semántico) sería sobreajustar a n=30 con la Spark mayormente caída — no se recomienda sin recalibrar con la Spark disponible.
- **Decisión secuencial (checkpoints 25s/40s) sin validar con pesos finales**: en la única corrida de calibración disponible, la decisión temprana nunca se disparó (0/30). No se sabe si dispara de forma razonable con los pesos que realmente corran en el juez.
- **Contrato del body sin confirmar**: se aceptan varios nombres de campo candidatos (ver [Contrato del endpoint](#contrato-del-endpoint)) y, si el body no es JSON reconocible, se intenta como base64 crudo directo — a falta de confirmación de los organizadores sobre el formato exacto que usa el evaluador real. Es una medida de máxima permisividad tomada por falta de tiempo para confirmar el schema, no un contrato validado.

## Estructura del repo

```
detect-altur/
├── app/                        # servicio (no tocado en esta limpieza)
├── calibracion/                # scripts y resultados de calibración, fuera del servicio
│   ├── calibrar.py
│   ├── calibrar_acustico_crudo.py
│   ├── calibrar_acustico_debug.py
│   ├── recalcular_acustico_todas.py
│   ├── calibracion_acustico_crudo.json
│   ├── calibracion_resultados_v2.jsonl
│   └── requirements-calibracion.txt
├── requirements.txt
├── .env.example
└── .gitignore
```
