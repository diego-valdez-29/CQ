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
- **Semántico** (`app/deteccion/semantico.py`): transcribe el canal del caller con Whisper y busca honestidad ante preguntas trampa (información que no existe). Si el transcript dice algo tipo "no sé"/"no tengo esa información" (regex), se resuelve como humano (`score=0.15`) sin llamar a ningún LLM. Si no, retorna directamente el score neutro (`score=0.5`) — **ya no llama a ningún LLM** (ver sección de calibración: la señal quedó con peso 0 en la fusión final). El cliente LLM (antes: la Spark, vía Tailscale) se conserva sin usar en `calibracion/semantico_llm_experimento.py`.

**Fusión**: promedio ponderado de las tres señales (`app/deteccion/fusion.py`). Los pesos en producción son los de la **calibración final** (ver siguiente sección): `{acustico: 0.0, comportamiento: 1.0, semantico: 0.0}` con umbral `0.56`.

**Decisión secuencial**: en vez de esperar la llamada completa, se evalúa en checkpoints de **25s** y **40s** de audio; si la confidence en un checkpoint cruza `UMBRAL_DECISION_TEMPRANA = 0.75`, se responde ahí sin esperar más. Si ningún checkpoint cruza el umbral, decide al final de la llamada.

## Estado honesto de calibración

**Calibración final** (post-fix del timeout de pared, ver historial de commits): muestra balanceada de **n=60 llamadas reales** (30 humanas / 30 sintéticas) del split `train` del dataset del reto (`manifest.csv`, 353 llamadas reales en total; `train` tiene 282). Pesos y umbral confirmados en **3 calibraciones independientes** con datos reales:

```
pesos  = {"acustico": 0.0, "comportamiento": 1.0, "semantico": 0.0}
umbral = 0.56
```

Con estos pesos, sobre la propia muestra de calibración:

- **Accuracy: 0.833**
- **EER: 0.183**
- **Matriz de confusión** `[[TN, FP], [FN, TP]] = [[22, 8], [2, 28]]`

Es decir: la señal que realmente decide es **comportamiento** — es la única que separó human/sintético de forma consistente y en la dirección correcta a través de las 3 corridas. `acústico` y `semántico` quedaron en **peso 0**, no porque no se probaran, sino porque se probaron con datos reales y no aportaron:

- **Semántico**: se probó tanto con el LLM de la Spark (vía Tailscale) como con OpenAI `gpt-4o-mini` (`calibracion/probar_semantico_openai.py`). Ninguno de los dos separó humano/sintético lo suficiente como para justificar la latencia y el riesgo de una llamada de red síncrona dentro de `/detect`. El código del cliente LLM se conservó, sin usar, en `calibracion/semantico_llm_experimento.py`. En producción, esta señal ahora solo corre el filtro regex de honestidad (gratis, sin red) y cae a un score neutro (`0.5`) si no resuelve — ver `app/deteccion/semantico.py`.
- **Acústico**: la heurística MFCC (`app/deteccion/acustico.py`) y la alternativa wav2vec2 (`calibracion/acustico_wav2vec.py`) se probaron ambas contra datos reales; ninguna separó de forma confiable en telefonía de 8kHz (ver `calibracion/calibracion_acustico_wav2vec.json` y `calibracion/calibracion_comportamiento_crudo.json`).

Con los pesos de decisión temprana calibrados (`UMBRAL_DECISION_TEMPRANA = 0.75`), el checkpoint temprano (25s/40s) se disparó en **3.3% de las llamadas** — comportamiento conservador y razonable: la decisión temprana existe pero no se dispara de forma agresiva, así que la mayoría de las llamadas se deciden con la señal completa.

En resumen: `acústico` y `semántico` son señales **exploradas y descartadas con evidencia real**, no señales sin intentar — el pipeline final es deliberadamente más simple (una sola señal decisiva) que el diseño original de tres señales ponderadas por igual.

## Cómo correrlo

```bash
cd detect-altur
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

El pipeline de producción (`app/main.py`) **no requiere ninguna variable de entorno**: la señal semántica ya no llama a ningún LLM (ver sección de calibración). `SPARK_API_KEY` (ver `.env.example`) solo la usa `calibracion/semantico_llm_experimento.py`, el cliente LLM conservado como referencia fuera del pipeline.

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

- **Una sola señal decisiva**: con `acustico` y `semantico` en peso 0, la clasificación final depende enteramente de `comportamiento`. Está calibrada y validada contra n=60 reales (accuracy 0.833, EER 0.183), pero es una sola fuente de evidencia — no hay redundancia si esa señal falla o se comporta distinto en el dataset real de los jueces.
- **Señal acústica descartada, no arreglada**: tanto la heurística MFCC (`app/deteccion/acustico.py`) como la alternativa wav2vec2 (`calibracion/acustico_wav2vec.py`) se probaron con datos reales y no separaron de forma confiable en telefonía de 8kHz. El código sigue corriendo en el pipeline (con peso 0, no afecta el resultado) por si se quiere reintentar más adelante con otro enfoque.
- **Señal semántica descartada, no arreglada**: se probó tanto el LLM de la Spark como OpenAI `gpt-4o-mini`; ninguno separó lo suficiente. El filtro regex de honestidad (paso 1) sigue activo porque es gratis y sin red, pero ya no hay paso 2 — el cliente LLM completo quedó en `calibracion/semantico_llm_experimento.py` por si se quiere retomar con otro prompt o modelo.
- **Calibración sobre n=60, no sobre el dataset completo de jueces**: los números de accuracy/EER/matriz de confusión son sobre la propia muestra de calibración (train, n=60 de 282 disponibles) — no hay garantía de que se mantengan igual sobre las llamadas reales del reto.
- **Contrato del body sin confirmar**: se aceptan varios nombres de campo candidatos (ver [Contrato del endpoint](#contrato-del-endpoint)) y, si el body no es JSON reconocible, se intenta como base64 crudo directo — a falta de confirmación de los organizadores sobre el formato exacto que usa el evaluador real. Es una medida de máxima permisividad tomada por falta de tiempo para confirmar el schema, no un contrato validado.

## Estructura del repo

```
detect-altur/
├── app/                        # servicio (no tocado en esta limpieza)
├── calibracion/                # scripts y resultados de calibración, fuera del servicio
│   ├── calibrar.py
│   ├── acustico_wav2vec.py           # señal acústica alternativa, probada y descartada
│   ├── semantico_llm_experimento.py  # cliente LLM (paso 2 de semántico), probado y descartado
│   ├── probar_semantico_openai.py    # prueba de semántico con OpenAI gpt-4o-mini
│   ├── calibracion_resultados.jsonl  # datos crudos de la calibración final (n=60)
│   └── requirements-calibracion.txt
├── requirements.txt
├── .env.example
└── .gitignore
```
