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

**Fusión**: promedio ponderado de las tres señales (`app/deteccion/fusion.py`). Los pesos en producción son los de la **calibración final** (ver siguiente sección): `{acustico: 0.0, comportamiento: 1.0, semantico: 0.0}` con umbral `0.58` (recalibrado sobre split `val`; el umbral original de la calibración en `train` era `0.56`).

**Decisión secuencial**: en vez de esperar la llamada completa, se evalúa en checkpoints de **25s** y **40s** de audio; si la confidence en un checkpoint cruza `UMBRAL_DECISION_TEMPRANA = 0.75`, se responde ahí sin esperar más. Si ningún checkpoint cruza el umbral, decide al final de la llamada.

## Estado honesto de calibración

**Calibración final** (post-fix del timeout de pared, ver historial de commits): muestra balanceada de **n=60 llamadas reales** (30 humanas / 30 sintéticas) del split `train` del dataset del reto (`manifest.csv`, 353 llamadas reales en total; `train` tiene 282). Pesos y umbral confirmados en **3 calibraciones independientes** con datos reales:

```
pesos  = {"acustico": 0.0, "comportamiento": 1.0, "semantico": 0.0}
umbral = 0.56  # calibrado sobre train; ver "Validación sobre split val" para el valor final (0.58)
```

Con estos pesos, sobre la propia muestra de calibración:

- **Accuracy: 0.833**
- **EER: 0.183**
- **Matriz de confusión** `[[TN, FP], [FN, TP]] = [[22, 8], [2, 28]]`

Es decir: la señal que realmente decide es **comportamiento** — es la única que separó human/sintético de forma consistente y en la dirección correcta a través de las 3 corridas. `acústico` y `semántico` quedaron en **peso 0**, no porque no se probaran, sino porque se probaron con datos reales y no aportaron:

- **Semántico**: se probó tanto con el LLM de la Spark (vía Tailscale) como con OpenAI `gpt-4o-mini` (`calibracion/probar_semantico_openai.py`). Ninguno de los dos separó humano/sintético lo suficiente como para justificar la latencia y el riesgo de una llamada de red síncrona dentro de `/detect`. Además, la transcripción con Whisper por sí sola — antes incluso de llegar al LLM — resultó ser el cuello de botella real: tomaba entre **25 y 90 segundos por llamada** (rango que terminó fijado como el timeout de pared adaptativo de Whisper, `WHISPER_TIMEOUT_MIN_S`/`WHISPER_TIMEOUT_MAX_S` en `app/deteccion/semantico.py`, precisamente porque así de lento era en la práctica), la causa principal de la latencia total de `/detect`. El código (Whisper, regex, cliente LLM) se conserva intacto en `app/deteccion/semantico.py` y en `calibracion/semantico_llm_experimento.py`.
- **Acústico**: la heurística MFCC (`app/deteccion/acustico.py`) y la alternativa wav2vec2 (`calibracion/acustico_wav2vec.py`) se probaron ambas contra datos reales; ninguna separó de forma confiable en telefonía de 8kHz (ver `calibracion/calibracion_acustico_wav2vec.json` y `calibracion/calibracion_comportamiento_crudo.json`).

Adicionalmente, se desplegó un modelo open-source self-hosted (`llama3.1:8b` vía Ollama, en infraestructura propia rentada por GPU-hora, sin depender de ningún proveedor comercial de IA) para descartar que el problema fuera específico de OpenAI o de la Spark (`calibracion/probar_semantico_vastai.py`). Se probó con dos semillas de muestreo distintas (n=16 cada una):

- Semilla 0: `|diff medias|=0.119`, dirección correcta (sintético > humano)
- Semilla 1: `|diff medias|=0.088`, dirección invertida (humano > sintético)

La inconsistencia de dirección entre semillas, con el mismo modelo y la misma arquitectura, confirma que el resultado favorable inicial fue ruido de muestra (n=16), no una señal real y estable. Con esta tercera fuente, la señal semántica fue validada contra **tres arquitecturas de LLM completamente distintas** — un modelo de razonamiento local (Spark), un modelo comercial en la nube (OpenAI), y un modelo open-source self-hosted (Ollama/Vast.ai) — y ninguna mostró separación confiable. Esto confirma que la limitación es inherente al contenido semántico de las llamadas del dataset (guiones cortos, poca oportunidad real de confabulación detectable), no una limitación de un proveedor específico.

Con el mismo estándar de evidencia, la señal acústica se llevó un paso más allá de la heurística y del modelo pre-entrenado: se entrenó un **clasificador de ML real** (no heurística) sobre datos reales del dataset (`calibracion/acustico_ml_entrenado.py`). Features MFCC (20 coeficientes) + delta + delta-delta extraídas del canal caller de las 282 llamadas de `train`, agregadas por llamada (media y varianza por coeficiente en el tiempo, no frame-by-frame, para no sobreajustar con tan pocos datos), entrenando una regresión logística (`StandardScaler` + `LogisticRegression`) y un Random Forest pequeño (50 árboles, `max_depth=4`). Validado sobre las 71 llamadas de `val` (nunca vistas en entrenamiento): **accuracy 0.972 (LR) y 0.958 (RF)** — muy por encima de `comportamiento` (0.789) y de la heurística MFCC original (0.141, prácticamente invertida).

Antes de creer estos números se verificaron explícitamente las explicaciones triviales de fuga de datos: split train/val hecho antes de cualquier ajuste (columna `split` del manifest original, no un split propio), `StandardScaler.fit()` corrido solo sobre `train` y aplicado a `val` solo con `.transform()` (confirmado línea por línea del código), sin sesgo de duración (146.5s vs 149.9s de media entre clases) ni de formato/sample rate (8kHz, PCM_16 en ambas clases) que explicara la separación por atajo, y sin un feature individual dominante en la importancia del Random Forest (máximo 0.125, repartido entre ~15 coeficientes de 120).

Descartada la fuga trivial, se sometió el modelo a la misma prueba de robustez ante degradación acústica que ya existe para `comportamiento` (`calibracion/estres_condiciones_ml_acustico.py`, mismas 10 llamadas de `val`, mismas 3 degradaciones — ver "Robustez ante degradación acústica" más abajo). El resultado es contundente: bajo ruido blanco (SNR~20dB), las **10 llamadas, sin excepción y sin importar su score original** (desde 0.009 hasta 1.000), colapsan al mismo score exacto `0.088` en la regresión logística — evidencia directa de que el modelo depende de un artefacto de banda muy estrecha del pipeline de generación/guardado de este dataset, no de una característica robusta de la síntesis de voz. Bajo códec agresivo (mp3 8kbps) y sub-muestreo extra — degradaciones mínimas que cualquier llamada telefónica real ya sufre — el modelo cambia de clasificación en hasta 2/10 casos, contra **0/10** de `comportamiento` en las mismas condiciones.

**Conclusión**: el 0.972/0.958 en val es sobreajuste al pipeline de generación de este dataset específico (*shortcut learning*), no una señal de síntesis de voz generalizable — el mismo patrón de falla ya observado con wav2vec2. No se integró a `fusion.py`. Código conservado en `calibracion/acustico_ml_entrenado.py` y `calibracion/estres_condiciones_ml_acustico.py` como exploración documentada.

Con los pesos de decisión temprana calibrados (`UMBRAL_DECISION_TEMPRANA = 0.75`), el checkpoint temprano (25s/40s) se disparó en **3.3% de las llamadas** — comportamiento conservador y razonable: la decisión temprana existe pero no se dispara de forma agresiva, así que la mayoría de las llamadas se deciden con la señal completa.

En resumen: `acústico` y `semántico` son señales **exploradas y descartadas con evidencia real**, no señales sin intentar — el pipeline final es deliberadamente más simple (una sola señal decisiva) que el diseño original de tres señales ponderadas por igual.

Con esto, la señal acústica fue explorada con **tres enfoques distintos** (heurística MFCC simple, modelo pre-entrenado wav2vec2, y ML entrenado propio sobre features MFCC+delta+delta2) — los tres fallaron la misma prueba de robustez ante degradación acústica, confirmando que el límite es estructural (información espectral insuficiente/no confiable en telefonía 8kHz de este dataset), no una limitación de un enfoque particular.

### Validación sobre split val (71 llamadas, nunca usadas en calibración)

El dataset del reto separa `manifest.csv` en `train` (282 llamadas) y `val` (71). Todas las calibraciones de arriba —pesos y umbral `0.56`— se hicieron exclusivamente sobre `train`; `val` nunca participó en ese ajuste. Correrlo sirve para saber si esos números sobreviven fuera de la muestra que los produjo (`calibracion/validar_val.py`).

Con los pesos fijos (`comportamiento=1.0`) y el umbral original de `train` (`0.56`), evaluando las 71 llamadas de `val` directamente contra `DetectorComportamiento` + `Fusion`:

- **Accuracy: 0.732** (vs. **0.833** en train — caída de 0.101, señal de cierto sobreajuste al split de calibración)
- **EER: 0.267** (vs. **0.183** en train)

Un barrido de umbral sobre las mismas 71 llamadas de `val` (0.30 a 0.70, paso 0.02) encontró que **0.58** maximiza accuracy en `val`:

| Umbral | Origen | Accuracy en val |
|---|---|---|
| 0.56 | calibrado en train | 0.732 |
| **0.58** | óptimo en val | **0.789** |

La brecha entre **0.833** (train) y **0.789** (val, ya con el umbral recalibrado) no desaparece del todo recalibrando el umbral, y es la comparación honesta a quedarse: el umbral se ajustó con datos de `val`, pero los pesos (`comportamiento=1.0`, el resto en 0) y la señal misma se calibraron enteramente en `train` — `val` nunca influyó en esa parte. Por eso `0.789` es una estimación más realista de accuracy en datos no vistos que `0.833`, que sí puede estar inflado por sobreajuste al propio split de calibración. Con esta evidencia se actualizó el umbral de producción de `0.56` a **`0.58`** en `app/main.py`.

Validación end-to-end vía HTTP real (`calibracion/validar_val_http.py`, POST real a `/detect` con las 71 llamadas de `val`, WAV codificado en base64, igual que un cliente real):

- **Latencia** (n=71, **0 fallos**): media **3.49s** · mediana **3.50s** · mínimo **0.52s** · p95 **4.52s** · máximo **5.13s**

Esto confirma en la práctica lo que predecía el diseño: sin Whisper en el camino (la causa raíz de la latencia anterior de 56-90s), `/detect` responde en el orden de segundos, medido con datos reales end-to-end y no solo estimado.

### Robustez ante degradación acústica (condiciones no vistas en calibración)

`calibracion/estres_condiciones.py` toma una muestra balanceada de n=10 llamadas de `val` (5 human / 5 synthetic) y prueba, sobre cada una, 3 degradaciones de audio que ni `train` ni `val` cubren: ruido blanco (SNR~20dB, solo canal caller), códec agresivo (mp3 a 8kbps y de vuelta a wav) y sub-muestreo extra (4kHz y de vuelta a 8kHz). Corre el mismo camino que producción (`DetectorComportamiento` + `Fusion`) sobre cada versión degradada y compara la clasificación contra el audio original:

| Degradación | Llamadas que cambiaron de clasificación |
|---|---|
| Códec agresivo (mp3 8kbps) | 0/10 |
| Sub-muestreo extra (4kHz↔8kHz) | 0/10 |
| Ruido blanco (SNR~20dB) | 4/10 |

`comportamiento` depende únicamente de `tiempo_primera_habla` (un timestamp, no del contenido espectral), así que códec y sub-muestreo — que degradan el espectro pero preservan la temporización del habla — no cambian nada. El ruido blanco sí cambia 4/10, pero no porque "confunda" el timing: en las **10** llamadas (no solo las 4), el ruido hace que silero-vad deje de detectar suficientes eventos de recuperación y el detector cae a su fallback conservador (`eventos_insuficientes`, `score=0.5`). Eso solo *cambia* la clasificación en las 4 que originalmente estaban por encima del umbral (0.58) — las otras 6 ya eran "human" antes y lo siguen siendo. Es decir: el sistema se degrada hacia "no sé" en vez de hacia una clasificación arriesgada.

### Calibración de confidence (Brier score, Platt scaling)

`confidence` (la salida numérica de `/detect`) es el score crudo de `comportamiento` ya fusionado — nunca se calibró como probabilidad. `calibracion/calibrar_confidence.py` mide qué tan bien calibrada está esa probabilidad sobre las 71 llamadas de `val` (Brier score: 0 = perfecto, 0.25 = tan bueno como predecir siempre 0.5) y prueba si Platt scaling (`sklearn.calibration._SigmoidCalibration`, ajuste 1D `calibrada = 1/(1+exp(a·score+b))`) lo mejora:

- **Brier score sin calibrar: 0.236** (cerca de 0.25 — la confidence cruda no refleja bien P(sintético))
- **Brier score con Platt scaling: 0.228** — mejora de **+3.5% relativo**
- **Decisión: no se aplicó.** El umbral considerado "mejora clara" es 5% relativo; 3.5% no lo cruza, y Platt scaling se ajustó y evaluó sobre las mismas 71 llamadas de `val` (sin split adicional para validar el ajuste), así que esa mejora ya es optimista in-sample — sobreajustar 2 parámetros más sobre el mismo conjunto no se justifica con esta ganancia.

Nota para si se retoma más adelante: el umbral de decisión (`0.58`) está calibrado sobre el score crudo, no sobre una escala calibrada — aplicar Platt scaling requeriría re-barrer el umbral sobre `confidence_calibrada`, no reusar `0.58` tal cual.

### Alcance real en producción

`app/main.py` **ya no instancia ni llama a `DetectorAcustico` ni a `DetectorSemantico`** — ninguno de los dos corre en el camino de `/detect`, ni siquiera el filtro regex gratuito de semántico. Solo se ejecuta `DetectorComportamiento` (peso 1.0 en `Fusion`). Las clases de ambos detectores descartados siguen intactas en `app/deteccion/acustico.py` y `app/deteccion/semantico.py` — documentadas como señales exploradas, no borradas — simplemente fuera del camino de evaluación.

El efecto directo es eliminar el costo de la transcripción con Whisper (los 25-90s por llamada de arriba) del tiempo de respuesta de `/detect`: `comportamiento` no depende de ningún modelo de ML, solo de la temporización de silencios/atropellos del audio, así que su costo es del orden de milisegundos por llamada. La medición end-to-end real sobre `/detect` (ver "Validación sobre split val" arriba: `calibracion/validar_val_http.py`, n=71, 0 fallos) confirma esto en la práctica: latencia media de **3.49s**, p95 de **4.52s**, máximo de **5.13s** — muy por debajo de los 25-90s que tomaba solo la transcripción con Whisper cuando `semántico` corría en el camino.

## Vulnerabilidad conocida: ataque de timing adversarial

`comportamiento` decide únicamente por `tiempo_primera_habla` (ver "Las tres señales"), y su media/std poblacional (`MEDIA_TIEMPO_PRIMERA_HABLA=8.82s`, `STD_TIEMPO_PRIMERA_HABLA=2.69s`) está hardcodeada en el código, no es secreta. Se midió qué tan explotable es esto, en 3 pasos, sobre una muestra fija de 10 llamadas `synthetic` de `val`:

1. **Ataque matemático** (`calibracion/experimento_adversarial.py`): de las 10, el detector clasificaba correctamente 5/10 hoy. Reemplazando `tiempo_primera_habla` por la media humana exacta (8.82s) y recalculando el mismo `_score_heuristico`, las 5 pasan de `is_synthetic=True` a `False` — **5/5 evaden**.

2. **Ataque real sobre audio** (`calibracion/ataque_audio_real.py`): en vez de simular el número, se modificó el WAV real de esas mismas 5 llamadas — recortando entre 1.58 y 3.98 segundos del silencio inicial (ya presente al comienzo de la llamada, sin sintetizar nada nuevo) del canal caller únicamente — para que Silero VAD detecte el primer habla cerca de 8.82s. Se re-corrió el pipeline completo (Silero VAD + `DetectorComportamiento` + `Fusion`) sobre el WAV modificado, leído de disco:
   - **5/5 evaden también con la manipulación real** (no solo en la simulación matemática).
   - Los 5 coinciden con la simulación matemática, y ninguno cae en el fallback `eventos_insuficientes` — es un engaño limpio del timing, no un artefacto de que la manipulación rompió la detección de eventos.

3. **Mitigación explorada — RMS del silencio previo** (`calibracion/detectar_silencio_artificial.py`): la hipótesis era que el silencio insertado por el ataque, al ser digitalmente perfecto (ceros), tendría un RMS mucho más bajo que el ruido de línea real que precede al habla en una llamada real, y que eso podría servir de chequeo adicional. **Descartada**: el ataque no inserta silencio artificial en estas 5 llamadas — solo recorta silencio real ya existente al inicio (los 5 deltas calculados dieron negativos), así que no hay ceros digitales que detectar. El RMS del silencio previo se solapa entre originales (`[0.00028, 0.01473]`) y atacadas (`[0.00026, 0.01921]`) — no hay un umbral simple de RMS que los separe.

Esto **no está mitigado** en `app/main.py` ni `fusion.py` — es una vulnerabilidad conocida y medida, no arreglada. Cualquier mitigación futura necesitaría una señal adicional independiente del timing (ver "Una sola señal decisiva" abajo), no un chequeo de energía sobre el silencio.

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

- **Una sola señal decisiva**: con `acustico` y `semantico` en peso 0, la clasificación final depende enteramente de `comportamiento`. Es una sola fuente de evidencia — no hay redundancia si esa señal falla o se comporta distinto en el dataset real de los jueces.
- **Vulnerable a un ataque de timing simple**: `comportamiento` depende únicamente de `tiempo_primera_habla`, cuya media/std poblacional está hardcodeada y no es secreta. Manipular el audio real (recortar el silencio inicial ya presente) para acercar ese timestamp a la media humana evade la detección en 5/5 casos probados, con audio real, no solo en simulación — y una mitigación simple por RMS del silencio no sirve (el ataque recorta silencio real, no inserta silencio digital). Ver "Vulnerabilidad conocida: ataque de timing adversarial". No mitigado todavía.
- **Accuracy real esperado más cercano a 0.79 que a 0.83**: los pesos y la señal se calibraron sobre train (n=60, accuracy 0.833, EER 0.183); sobre val (n=71, nunca usado en esa calibración) la accuracy cae a 0.732 con el umbral original, señal de cierto sobreajuste al split de calibración. Recalibrar solo el umbral (0.56 → 0.58, el valor ya en producción) recupera parte de esa caída (accuracy 0.789 en val), pero no toda — ver "Validación sobre split val". `0.789` es la estimación más honesta de accuracy en datos no vistos.
- **Señal acústica descartada con tres enfoques distintos, no arreglada**: la heurística MFCC (`app/deteccion/acustico.py`), la alternativa wav2vec2 (`calibracion/acustico_wav2vec.py`) y un clasificador de ML propio entrenado sobre features MFCC+delta+delta2 (`calibracion/acustico_ml_entrenado.py`, accuracy 0.972 en val) se probaron con datos reales. El ML propio parecía prometedor hasta someterlo a la misma prueba de robustez ante degradación acústica que ya se aplica a `comportamiento` (`calibracion/estres_condiciones_ml_acustico.py`): ante ruido blanco (SNR~20dB) las 10 llamadas de prueba colapsan al mismo score exacto sin importar su valor original, y ante códec/sub-muestreo mínimos cambia clasificación en hasta 2/10 casos (vs. 0/10 de `comportamiento`) — evidencia de sobreajuste al pipeline de generación del dataset (*shortcut learning*), no de una señal robusta de síntesis de voz. El código de los tres enfoques sigue corriendo o disponible (con peso 0 en producción, no afecta el resultado) por si se quiere reintentar más adelante con otro enfoque o dataset.
- **Señal semántica descartada, no arreglada**: se probó tanto el LLM de la Spark como OpenAI `gpt-4o-mini`; ninguno separó lo suficiente. El filtro regex de honestidad (paso 1) sigue activo porque es gratis y sin red, pero ya no hay paso 2 — el cliente LLM completo quedó en `calibracion/semantico_llm_experimento.py` por si se quiere retomar con otro prompt o modelo.
- **Umbral recalibrado sobre val, pero solo el umbral**: el barrido de `0.58` se hizo sobre las mismas 71 llamadas de `val` que ahora sirven de "conjunto de prueba" — técnicamente ese umbral ya no es 100% independiente de los datos que se usan para reportarlo. Los pesos y la señal siguen calibrados solo en train, que es donde está el sobreajuste real.
- **Contrato del body sin confirmar**: se aceptan varios nombres de campo candidatos (ver [Contrato del endpoint](#contrato-del-endpoint)) y, si el body no es JSON reconocible, se intenta como base64 crudo directo — a falta de confirmación de los organizadores sobre el formato exacto que usa el evaluador real. Es una medida de máxima permisividad tomada por falta de tiempo para confirmar el schema, no un contrato validado.

## Estructura del repo

```
detect-altur/
├── app/                        # servicio (no tocado en esta limpieza)
├── calibracion/                # scripts y resultados de calibración, fuera del servicio
│   ├── calibrar.py
│   ├── acustico_wav2vec.py           # señal acústica alternativa, probada y descartada
│   ├── acustico_ml_entrenado.py      # ML propio sobre MFCC+delta+delta2, probado y descartado
│   ├── estres_condiciones_ml_acustico.py  # prueba de robustez que descartó acustico_ml_entrenado.py
│   ├── semantico_llm_experimento.py  # cliente LLM (paso 2 de semántico), probado y descartado
│   ├── probar_semantico_openai.py    # prueba de semántico con OpenAI gpt-4o-mini
│   ├── calibracion_resultados.jsonl  # datos crudos de la calibración final (n=60)
│   └── requirements-calibracion.txt
├── requirements.txt
├── .env.example
└── .gitignore
```
