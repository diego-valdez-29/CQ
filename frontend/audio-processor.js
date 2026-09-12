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
   * @param {Float32Array} samples Muestras de audio normalizadas [-1.0, 1.0]
   * @param {number} sampleRate Tasa de muestreo (8000)
   * @param {number} numChannels Número de canales (1 para mono, 2 para estéreo)
   * @returns {ArrayBuffer}
   */
  encodeWAV(samples, sampleRate = 8000, numChannels = 1) {
    const bytesPerSample = 2; // 16 bits
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
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

    // Escribir muestras PCM de 16 bits con recorte (clipping) suave
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      // Convertir de [-1.0, 1.0] a entero con signo de 16 bits [-32768, 32767]
      const val = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(offset, val, true);
      offset += 2;
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

  /**
   * Procesa un archivo de audio seleccionado por el usuario (WAV/MP3/OGG) y lo convierte a 8kHz WAV Base64
   * @param {File|Blob} file Archivo de audio
   * @returns {Promise<{ base64: string, durationSeconds: number, blob: Blob, sampleRate: number }>}
   */
  async processAudioFile(file) {
    const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtxClass();
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    // Obtener datos del Canal 0 (caller / receptor)
    const channel0Data = audioBuffer.getChannelData(0);
    const resampled = this.resampleAudio(channel0Data, audioBuffer.sampleRate, this.targetSampleRate);
    const wavBuffer = this.encodeWAV(resampled, this.targetSampleRate, 1);
    const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });
    const base64 = await this.arrayBufferToBase64(wavBuffer);
    const durationSeconds = resampled.length / this.targetSampleRate;

    await ctx.close();

    return {
      base64,
      durationSeconds,
      blob: wavBlob,
      sampleRate: this.targetSampleRate,
      channels: 1,
      totalSamples: resampled.length,
    };
  }
}

// Exportar como objeto global
window.AudioProcessor = AudioProcessor;
