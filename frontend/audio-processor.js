/**
 * audio-processor.js
 * Módulo de captura, remuestreo a 8kHz, codificación WAV PCM 16-bit y conversión a Base64.
 * Cumple con el estándar de telefonía de 8000 Hz requerido por la API de Altur.
 */

class AudioProcessor {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.scriptProcessor = null;
    this.analyser = null;
    this.recordedSamples = [];
    this.isRecording = false;
    this.sampleRateInput = 44100;
    this.targetSampleRate = 8000; // 8kHz exigido por el detector

    // --- Llamada Simulada (canal 1 = agente de referencia) ---
    this.agentReferenceUrl = "assets/agente_referencia.wav";
    this.agentReferenceSamples = null; // Float32Array a targetSampleRate
    this.agentReferenceDurationSeconds = 0;
  }

  /**
   * Inicia la captura desde el micrófono del usuario
   * @param {Function} onAudioProcess Callback opcional para monitorear muestras
   */
  async startRecording(onAudioProcess = null) {
    if (this.isRecording) return;

    // Crear o reanudar AudioContext
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioCtxClass();
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    this.sampleRateInput = this.audioContext.sampleRate;

    // Solicitar permiso de micrófono
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    // Nodo Analyser para el visualizador
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.8;
    this.sourceNode.connect(this.analyser);

    // Buffer de captura
    this.recordedSamples = [];
    const bufferSize = 4096;
    this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    this.scriptProcessor.onaudioprocess = (event) => {
      if (!this.isRecording) return;
      const inputChannel = event.inputBuffer.getChannelData(0);
      // Copiar muestras Float32
      const chunk = new Float32Array(inputChannel);
      this.recordedSamples.push(chunk);

      if (typeof onAudioProcess === 'function') {
        onAudioProcess(chunk);
      }
    };

    this.sourceNode.connect(this.scriptProcessor);
    this.scriptProcessor.connect(this.audioContext.destination);

    this.isRecording = true;
  }

  /**
   * Detiene la captura y devuelve el audio codificado en WAV a 8kHz y Base64
   * @returns {Promise<{ base64: string, durationSeconds: number, blob: Blob, sampleRate: number, channels: number }>}
   */
  async stopRecording() {
    if (!this.isRecording) {
      throw new Error("No hay una grabación activa");
    }

    this.isRecording = false;

    // Desconectar nodos de audio
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    // Concatenar todos los chunks en un único Float32Array
    const totalLength = this.recordedSamples.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.recordedSamples) {
      mergedSamples.set(chunk, offset);
      offset += chunk.length;
    }
    this.recordedSamples = [];

    // Remuestrear a 8000 Hz
    const resampled = this.resampleAudio(mergedSamples, this.sampleRateInput, this.targetSampleRate);

    // Codificar a WAV 16-bit PCM
    const wavBuffer = this.encodeWAV(resampled, this.targetSampleRate, 1);
    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
    const base64 = await this.arrayBufferToBase64(wavBuffer);
    const durationSeconds = resampled.length / this.targetSampleRate;

    return {
      base64,
      durationSeconds,
      blob: wavBlob,
      sampleRate: this.targetSampleRate,
      channels: 1,
      totalSamples: resampled.length,
    };
  }

  /**
   * Remuestrea un arreglo de muestras Float32Array de un sampleRate a otro usando interpolación lineal de alta calidad
   */
  resampleAudio(audioBuffer, fromSampleRate, toSampleRate) {
    if (fromSampleRate === toSampleRate) {
      return audioBuffer;
    }

    const ratio = fromSampleRate / toSampleRate;
    const newLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const originalIndex = i * ratio;
      const indexPrev = Math.floor(originalIndex);
      const indexNext = Math.min(indexPrev + 1, audioBuffer.length - 1);
      const fraction = originalIndex - indexPrev;

      // Interpolación lineal
      result[i] = audioBuffer[indexPrev] * (1 - fraction) + audioBuffer[indexNext] * fraction;
    }

    return result;
  }

  /**
   * Construye un archivo WAV estándar RIFF de 16 bits PCM
   * @param {Float32Array|Float32Array[]} samples Muestras normalizadas [-1.0, 1.0]:
   *   un solo Float32Array para mono, o [canalIzq, canalDer] para estéreo (numChannels=2)
   * @param {number} sampleRate Tasa de muestreo (8000)
   * @param {number} numChannels Número de canales (1 para mono, 2 para estéreo)
   * @returns {ArrayBuffer}
   */
  encodeWAV(samples, sampleRate = 8000, numChannels = 1) {
    const bytesPerSample = 2; // 16 bits
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;

    let frameCount;
    let getSample;

    if (numChannels === 2 && Array.isArray(samples)) {
      const [left, right] = samples;
      frameCount = Math.min(left.length, right.length);
      getSample = (i, ch) => (ch === 0 ? left[i] : right[i]);
    } else {
      frameCount = samples.length;
      getSample = (i) => samples[i];
    }

    const dataSize = frameCount * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    /* RIFF header */
    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true); // Tamaño total menos 8 bytes
    this.writeString(view, 8, "WAVE");

    /* "fmt " sub-chunk */
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // Tamaño del sub-chunk fmt (16 para PCM)
    view.setUint16(20, 1, true); // Formato de audio (1 = PCM lineal)
    view.setUint16(22, numChannels, true); // Número de canales
    view.setUint32(24, sampleRate, true); // Sample rate (8000 Hz)
    view.setUint32(28, byteRate, true); // Byte rate
    view.setUint16(32, blockAlign, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample (16)

    /* "data" sub-chunk */
    this.writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    // Escribir muestras PCM de 16 bits con recorte (clipping) suave, entrelazadas por frame
    let offset = 44;
    for (let i = 0; i < frameCount; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        let s = Math.max(-1, Math.min(1, getSample(i, ch)));
        // Convertir de [-1.0, 1.0] a entero con signo de 16 bits [-32768, 32767]
        const val = s < 0 ? s * 0x8000 : s * 0x7fff;
        view.setInt16(offset, val, true);
        offset += 2;
      }
    }

    return buffer;
  }

  /**
   * Escribe una cadena ASCII en una vista DataView
   */
  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  /**
   * Convierte un ArrayBuffer a cadena Base64 en bloques seguros
   */
  async arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const len = bytes.byteLength;
    const chunkSize = 0x8000; // 32KB chunks para evitar límite de argumentos en String.fromCharCode

    for (let i = 0; i < len; i += chunkSize) {
      const sub = bytes.subarray(i, Math.min(i + chunkSize, len));
      binary += String.fromCharCode.apply(null, sub);
    }
    return window.btoa(binary);
  }

  // =========================================================================
  // Llamada Simulada: reconstruye una estructura real de 2 canales
  // (canal 0 = caller a evaluar, canal 1 = agente de referencia) tanto si el
  // canal 0 viene del micrófono real como de un archivo TTS ya generado.
  // =========================================================================

  /**
   * Carga (una sola vez) el audio de referencia del agente (assets/agente_referencia.wav),
   * lo remuestrea a targetSampleRate y devuelve su duración exacta en segundos.
   * @returns {Promise<number>} Duración en segundos del audio de referencia
   */
  async loadAgentReference() {
    if (this.agentReferenceSamples) {
      return this.agentReferenceDurationSeconds;
    }

    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    const decodeCtx = new AudioCtxClass();

    try {
      const res = await fetch(this.agentReferenceUrl);
      if (!res.ok) {
        throw new Error(`No se pudo cargar ${this.agentReferenceUrl} (HTTP ${res.status})`);
      }
      const arrayBuffer = await res.arrayBuffer();
      const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);

      this.agentReferenceSamples = this.resampleAudio(
        audioBuffer.getChannelData(0),
        audioBuffer.sampleRate,
        this.targetSampleRate
      );
      this.agentReferenceDurationSeconds = this.agentReferenceSamples.length / this.targetSampleRate;

      return this.agentReferenceDurationSeconds;
    } finally {
      await decodeCtx.close();
    }
  }

  /**
   * Decodifica un archivo de audio (p. ej. un TTS ya generado) para usarlo directamente como
   * canal 0 (caller), sin grabar nada en vivo. Remuestrea a targetSampleRate.
   * @param {File|Blob} file
   * @returns {Promise<Float32Array>}
   */
  async decodeCallerFile(file) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    const decodeCtx = new AudioCtxClass();

    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
      return this.resampleAudio(audioBuffer.getChannelData(0), audioBuffer.sampleRate, this.targetSampleRate);
    } finally {
      await decodeCtx.close();
    }
  }

  /**
   * Espera `durationSeconds` usando AudioContext.currentTime como reloj de referencia
   * (en vez de Date.now/setTimeout) para anclar con precisión el t=0 del turno del agente.
   * Invoca onTick(elapsedSeconds) periódicamente para animar un timer visual.
   * @param {number} durationSeconds
   * @param {Function} onTick
   */
  async waitForDuration(durationSeconds, onTick = null) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    if (!this.audioContext) {
      this.audioContext = new AudioCtxClass();
    }
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    const ctx = this.audioContext;
    const startTime = ctx.currentTime;

    return new Promise((resolve) => {
      const tick = () => {
        const elapsed = ctx.currentTime - startTime;
        if (typeof onTick === "function") {
          onTick(Math.min(elapsed, durationSeconds));
        }
        if (elapsed >= durationSeconds) {
          resolve();
        } else {
          requestAnimationFrame(tick);
        }
      };
      tick();
    });
  }

  /**
   * Inicia la captura real de micrófono para el turno del caller dentro de una Llamada Simulada.
   * Reutiliza el mismo mecanismo de captura que startRecording (getUserMedia + ScriptProcessor),
   * ligado al AudioContext ya usado como reloj de referencia por waitForDuration.
   * @param {Function} onAudioProcess Callback opcional para el visualizador
   */
  async startCallerMicCapture(onAudioProcess = null) {
    return this.startRecording(onAudioProcess);
  }

  /**
   * Detiene la captura de micrófono del turno del caller y devuelve las muestras
   * remuestreadas a targetSampleRate SIN codificarlas todavía a WAV (para poder
   * combinarlas después con el canal del agente en buildSimulatedCallWav).
   * @returns {Promise<Float32Array>}
   */
  async stopCallerMicCapture() {
    if (!this.isRecording) {
      throw new Error("No hay una grabación activa");
    }

    this.isRecording = false;

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    const totalLength = this.recordedSamples.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.recordedSamples) {
      mergedSamples.set(chunk, offset);
      offset += chunk.length;
    }
    this.recordedSamples = [];

    return this.resampleAudio(mergedSamples, this.sampleRateInput, this.targetSampleRate);
  }

  /**
   * Combina el canal del caller (mic o TTS, ya a targetSampleRate) con el canal del agente
   * de referencia en un único WAV estéreo real:
   *   canal 0 = silencio durante el turno del agente + contenido real del caller
   *   canal 1 = agente de referencia + silencio (si el caller dura más que el agente)
   * Nunca recorta el canal más largo -- rellena el más corto con ceros.
   *
   * EXPERIMENTAL, no representa el caso de uso real del reto: esta estructura es
   * secuencial (agente habla todo su turno completo, y solo DESPUÉS arranca el
   * turno del caller), no intercalada como una llamada real. Se probó contra
   * /detect en vivo (agente=agente_referencia.wav, caller=canal 0 de una llamada
   * synthetic real del dataset) y el resultado fue confidence=0.5/is_synthetic=false
   * -- el fallback neutro de DetectorComportamiento, no una clasificación real.
   *
   * Motivo (ver app/deteccion/comportamiento.py, DetectorComportamiento.analizar):
   * la señal en producción (peso 1.0) solo calcula un score real si encuentra >=2
   * "eventos de recuperación" -- una pausa/atropello del agente seguido de habla
   * del caller dentro de una ventana de 5s. Con este layout secuencial, TODAS las
   * pausas del agente ocurren en [0, duración_agente) y el caller no habla hasta
   * exactamente duración_agente, así que como máximo UN evento (el más cercano al
   * corte) puede caer dentro de esa ventana de 5s -- nunca dos. En la práctica esto
   * dispara casi siempre el fallback "eventos_insuficientes" (score=0.5), sin
   * importar si el audio del caller es humano o sintético. Confirmado llamando a
   * DetectorComportamiento directamente: 10 eventos agente detectados, 0
   * recuperaciones válidas, contra la misma llamada original sin modificar (con
   * turnos intercalados reales) que sí clasifica correctamente is_synthetic=true.
   *
   * Un fix real requeriría intercalar el audio del caller en las pausas reales
   * del agente (turno por turno, con prompts sincronizados a los timestamps de
   * pausa detectados), no una única grabación continua después de un timer.
   * Se deja así a propósito -- ver la nota "Experimental" en index.html.
   *
   * @param {Float32Array} callerSamples Audio del caller (mic o TTS), ya a targetSampleRate
   * @returns {{ base64: Promise<string>, wavBuffer: ArrayBuffer, blob: Blob, durationSeconds: number, totalSamples: number }}
   */
  buildSimulatedCallWav(callerSamples) {
    if (!this.agentReferenceSamples) {
      throw new Error("El audio de referencia del agente no está cargado (llama a loadAgentReference primero)");
    }
    const agentSamples = this.agentReferenceSamples;

    // Canal 0: silencio (ceros) durante el turno del agente, luego el audio real del caller
    const callerTrack = new Float32Array(agentSamples.length + callerSamples.length);
    callerTrack.set(callerSamples, agentSamples.length);

    const totalLength = Math.max(callerTrack.length, agentSamples.length);
    const left = new Float32Array(totalLength); // canal 0 = caller
    const right = new Float32Array(totalLength); // canal 1 = agente
    left.set(callerTrack, 0);
    right.set(agentSamples, 0);

    const wavBuffer = this.encodeWAV([left, right], this.targetSampleRate, 2);
    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
    const durationSeconds = totalLength / this.targetSampleRate;

    return {
      wavBuffer,
      blob: wavBlob,
      durationSeconds,
      totalSamples: totalLength,
    };
  }

  /**
   * Procesa un archivo de audio seleccionado por el usuario (WAV/MP3/OGG) y lo convierte a 8kHz WAV Base64.
   * Si el archivo de origen es estéreo, preserva ambos canales (canal 0 = caller, canal 1 = agente)
   * en vez de colapsar a mono.
   * @param {File|Blob} file Archivo de audio
   * @returns {Promise<{ base64: string, durationSeconds: number, blob: Blob, sampleRate: number, channels: number }>}
   */
  async processAudioFile(file) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtxClass();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    const isStereo = audioBuffer.numberOfChannels >= 2;
    let wavBuffer, totalSamples;

    if (isStereo) {
      const left = this.resampleAudio(audioBuffer.getChannelData(0), audioBuffer.sampleRate, this.targetSampleRate);
      const right = this.resampleAudio(audioBuffer.getChannelData(1), audioBuffer.sampleRate, this.targetSampleRate);
      wavBuffer = this.encodeWAV([left, right], this.targetSampleRate, 2);
      totalSamples = Math.min(left.length, right.length);
    } else {
      const resampled = this.resampleAudio(audioBuffer.getChannelData(0), audioBuffer.sampleRate, this.targetSampleRate);
      wavBuffer = this.encodeWAV(resampled, this.targetSampleRate, 1);
      totalSamples = resampled.length;
    }

    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
    const base64 = await this.arrayBufferToBase64(wavBuffer);
    const durationSeconds = totalSamples / this.targetSampleRate;

    await ctx.close();

    return {
      base64,
      durationSeconds,
      blob: wavBlob,
      sampleRate: this.targetSampleRate,
      channels: isStereo ? 2 : 1,
      totalSamples,
    };
  }
}

// Exportar como objeto global
window.AudioProcessor = AudioProcessor;
