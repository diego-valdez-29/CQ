/**
 * app.js
 * Orquestador principal de la interfaz frontend de Detección de Voz IA (Altur).
 * Conecta captura de audio en streaming, procesamiento a 8kHz, envío a /detect con channel=0
 * y visualización interactiva de resultados (is_synthetic, confidence).
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // 1. Elementos del DOM y Referencias
  // =========================================================================
  const elements = {
    // Navegación y Conectividad
    apiStatusDot: document.getElementById('apiStatusDot'),
    apiStatusText: document.getElementById('apiStatusText'),
    btnConfigToggle: document.getElementById('btnConfigToggle'),
    configDrawer: document.getElementById('configDrawer'),
    inputApiUrl: document.getElementById('inputApiUrl'),
    selectCorsMode: document.getElementById('selectCorsMode'),
    selectCorsCredentials: document.getElementById('selectCorsCredentials'),
    inputChannel: document.getElementById('inputChannel'),
    btnSaveConfig: document.getElementById('btnSaveConfig'),

    // Modos y Visualizador
    tabMic: document.getElementById('tabMic'),
    tabFile: document.getElementById('tabFile'),
    fileDropArea: document.getElementById('fileDropArea'),
    fileInput: document.getElementById('fileInput'),
    canvas: document.getElementById('visualizerCanvas'),
    recordingTimer: document.getElementById('recordingTimer'),
    timerText: document.getElementById('timerText'),
    timerRecDot: document.getElementById('timerRecDot'),
    audioRateBadge: document.getElementById('audioRateBadge'),

    // Botones de Acción
    btnRecord: document.getElementById('btnRecord'),
    btnRecordText: document.getElementById('btnRecordText'),
    btnSampleAudio: document.getElementById('btnSampleAudio'),
    btnUploadWav: document.getElementById('btnUploadWav'),

    // Panel de Resultados
    resultIdle: document.getElementById('resultIdle'),
    resultAnalyzing: document.getElementById('resultAnalyzing'),
    resultActive: document.getElementById('resultActive'),
    verdictCard: document.getElementById('verdictCard'),
    verdictIcon: document.getElementById('verdictIcon'),
    verdictLabel: document.getElementById('verdictLabel'),
    verdictTitle: document.getElementById('verdictTitle'),
    verdictDesc: document.getElementById('verdictDesc'),

    // Medidor de Confianza
    gaugeCircle: document.getElementById('gaugeCircle'),
    gaugeText: document.getElementById('gaugeText'),
    gaugeTierPill: document.getElementById('gaugeTierPill'),
    progressBarFill: document.getElementById('progressBarFill'),

    // Telemetría y Audio Playback
    metricChannel: document.getElementById('metricChannel'),
    metricRate: document.getElementById('metricRate'),
    metricLatency: document.getElementById('metricLatency'),
    metricSize: document.getElementById('metricSize'),
    audioPlayer: document.getElementById('audioPlayer'),
    jsonResponsePre: document.getElementById('jsonResponsePre'),
    btnCopyJson: document.getElementById('btnCopyJson'),

    // Historial
    historyTableBody: document.getElementById('historyTableBody'),
    historyEmptyRow: document.getElementById('historyEmptyRow'),
    btnExportHistory: document.getElementById('btnExportHistory'),
    btnClearHistory: document.getElementById('btnClearHistory'),
  };

  // =========================================================================
  // 2. Estado de la Aplicación
  // =========================================================================
  // Detectar URL inicial: si se sirve desde FastAPI usar el mismo origen, si no https://api-como-quieras.devs-dom.com/detect
  const isServedFromBackend = window.location.port === '8000' || window.location.hostname === 'localhost';
  const defaultApiUrl = window.location.origin.includes('http') && window.location.port === '8000'
    ? `${window.location.origin}/detect`
    : 'http://localhost:8000/detect';

  const state = {
    apiUrl: localStorage.getItem('altur_api_url') || defaultApiUrl,
    corsMode: localStorage.getItem('altur_cors_mode') || 'cors',
    corsCredentials: localStorage.getItem('altur_cors_credentials') || 'omit',
    channel: 0, // Canal 0 fijo para análisis del receptor/interlocutor
    isRecording: false,
    recordStartTime: 0,
    timerInterval: null,
    audioProcessor: new AudioProcessor(),
    visualizer: new AudioVisualizer('visualizerCanvas'),
    lastResult: null,
    lastAudioBlob: null,
    history: [],
  };

  elements.inputApiUrl.value = state.apiUrl;
  elements.inputChannel.value = state.channel;
  if (elements.selectCorsMode) elements.selectCorsMode.value = state.corsMode;
  if (elements.selectCorsCredentials) elements.selectCorsCredentials.value = state.corsCredentials;

  // =========================================================================
  // 3. Verificación de Salud de la API (Health Check)
  // =========================================================================
  async function checkApiHealth() {
    elements.apiStatusDot.className = 'status-indicator-dot';
    elements.apiStatusText.textContent = 'Verificando...';

    try {
      // Extraer base URL
      const urlObj = new URL(state.apiUrl);
      const healthUrl = `${urlObj.origin}/health`;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3500);

      const res = await fetch(healthUrl, {
        method: 'GET',
        mode: state.corsMode || 'cors',
        credentials: state.corsCredentials || 'omit',
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        elements.apiStatusDot.className = 'status-indicator-dot online';
        elements.apiStatusText.textContent = `Online (${urlObj.hostname})`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      // Intentar ping con método OPTIONS sobre /detect con modo CORS
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2500);
        await fetch(state.apiUrl, {
          method: 'OPTIONS',
          mode: state.corsMode || 'cors',
          credentials: state.corsCredentials || 'omit',
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        elements.apiStatusDot.className = 'status-indicator-dot online';
        elements.apiStatusText.textContent = 'Online (CORS OK)';
      } catch (e) {
        elements.apiStatusDot.className = 'status-indicator-dot offline';
        elements.apiStatusText.textContent = 'Desconectado / CORS';
      }
    }
  }

  checkApiHealth();
  setInterval(checkApiHealth, 15000);

  // =========================================================================
  // 4. Configuración y Drawer
  // =========================================================================
  elements.btnConfigToggle.addEventListener('click', () => {
    elements.configDrawer.classList.toggle('open');
  });

  elements.btnSaveConfig.addEventListener('click', () => {
    const newUrl = elements.inputApiUrl.value.trim();
    const newCorsMode = elements.selectCorsMode ? elements.selectCorsMode.value : 'cors';
    const newCorsCredentials = elements.selectCorsCredentials ? elements.selectCorsCredentials.value : 'omit';

    if (newUrl) {
      state.apiUrl = newUrl;
      state.corsMode = newCorsMode;
      state.corsCredentials = newCorsCredentials;

      localStorage.setItem('altur_api_url', newUrl);
      localStorage.setItem('altur_cors_mode', newCorsMode);
      localStorage.setItem('altur_cors_credentials', newCorsCredentials);

      elements.configDrawer.classList.remove('open');
      checkApiHealth();
    }
  });

  // =========================================================================
  // 5. Pestañas de Modo (Micrófono vs Archivo)
  // =========================================================================
  elements.tabMic.addEventListener('click', () => {
    elements.tabMic.classList.add('active');
    elements.tabFile.classList.remove('active');
    elements.fileDropArea.classList.remove('active');
  });

  elements.tabFile.addEventListener('click', () => {
    elements.tabFile.classList.add('active');
    elements.tabMic.classList.remove('active');
    elements.fileDropArea.classList.add('active');
  });

  // =========================================================================
  // 6. Grabación de Micrófono en Vivo
  // =========================================================================
  elements.btnRecord.addEventListener('click', async () => {
    if (!state.isRecording) {
      await startLiveRecording();
    } else {
      await stopLiveRecordingAndAnalyze();
    }
  });

  async function startLiveRecording() {
    try {
      await state.audioProcessor.startRecording();
      state.isRecording = true;
      state.recordStartTime = Date.now();

      // Conectar visualizador al analyser
      state.visualizer.connect(state.audioProcessor.analyser);

      // Actualizar UI
      elements.btnRecord.classList.add('recording');
      elements.btnRecordText.textContent = 'Detener y Analizar Llamada';
      elements.timerRecDot.classList.add('active');

      // Iniciar timer
      elements.timerText.textContent = '00:00';
      state.timerInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - state.recordStartTime) / 1000);
        const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const secs = String(elapsedSec % 60).padStart(2, '0');
        elements.timerText.textContent = `${mins}:${secs}`;
      }, 500);

    } catch (err) {
      console.error('Error al iniciar grabación:', err);
      alert(`No se pudo acceder al micrófono: ${err.message}. Asegúrate de conceder permisos de audio.`);
    }
  }

  async function stopLiveRecordingAndAnalyze() {
    if (!state.isRecording) return;

    // Detener timer
    clearInterval(state.timerInterval);
    elements.timerRecDot.classList.remove('active');
    elements.btnRecord.classList.remove('recording');
    elements.btnRecordText.textContent = 'Procesando Audio...';

    // Desconectar visualizador
    state.visualizer.disconnect();

    try {
      // Obtener audio remuestreado a 8kHz y Base64
      const audioData = await state.audioProcessor.stopRecording();
      state.isRecording = false;
      elements.btnRecordText.textContent = 'Iniciar Grabación de Llamada';

      if (audioData.durationSeconds < 0.5) {
        alert('La grabación es demasiado corta. Graba al menos 1 o 2 segundos de voz para analizar.');
        return;
      }

      await sendAudioForDetection(audioData.base64, audioData.blob, audioData.durationSeconds, 'Micrófono en vivo');
    } catch (err) {
      state.isRecording = false;
      elements.btnRecordText.textContent = 'Iniciar Grabación de Llamada';
      console.error('Error al procesar audio:', err);
      alert(`Error al procesar el audio: ${err.message}`);
    }
  }

  // =========================================================================
  // 7. Subida y Procesamiento de Archivos WAV
  // =========================================================================
  elements.btnUploadWav.addEventListener('click', () => {
    elements.fileInput.click();
  });

  elements.fileDropArea.addEventListener('click', () => {
    elements.fileInput.click();
  });

  elements.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      await processSelectedFile(file);
    }
  });

  // Drag and Drop
  elements.fileDropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.fileDropArea.classList.add('dragover');
  });

  elements.fileDropArea.addEventListener('dragleave', () => {
    elements.fileDropArea.classList.remove('dragover');
  });

  elements.fileDropArea.addEventListener('drop', async (e) => {
    e.preventDefault();
    elements.fileDropArea.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processSelectedFile(e.dataTransfer.files[0]);
    }
  });

  async function processSelectedFile(file) {
    showAnalyzingState(`Procesando ${file.name} y remuestreando a 8kHz...`);
    try {
      const audioData = await state.audioProcessor.processAudioFile(file);
      await sendAudioForDetection(audioData.base64, audioData.blob, audioData.durationSeconds, file.name);
    } catch (err) {
      showIdleState();
      console.error('Error al procesar archivo:', err);
      alert(`Error al decodificar el archivo de audio: ${err.message}`);
    }
  }

  // =========================================================================
  // 8. Botón de Prueba Rápida con Muestra (audios/prueba.wav o Demo)
  // =========================================================================
  elements.btnSampleAudio.addEventListener('click', async () => {
    showAnalyzingState('Cargando y analizando audio de muestra (prueba.wav)...');
    try {
      // Intentar cargar prueba.wav desde el directorio local o audios/
      let blob = null;
      for (const path of ['prueba.wav', '../audios/prueba.wav', '/static/prueba.wav']) {
        try {
          const res = await fetch(path);
          if (res.ok) {
            blob = await res.blob();
            break;
          }
        } catch (e) { }
      }

      if (!blob) {
        // Fallback: generar tono sintético telefónico a 8kHz
        blob = generateSyntheticSampleBlob();
      }

      const audioData = await state.audioProcessor.processAudioFile(blob);
      await sendAudioForDetection(audioData.base64, audioData.blob, audioData.durationSeconds, 'prueba.wav (Muestra)');
    } catch (err) {
      showIdleState();
      console.error('Error con audio de muestra:', err);
      alert(`No se pudo cargar el audio de muestra: ${err.message}`);
    }
  });

  function generateSyntheticSampleBlob() {
    // Generar tono telefónico multifrecuencia / voz sintética de 3 segundos a 8kHz
    const sampleRate = 8000;
    const duration = 3.0;
    const numSamples = Math.floor(sampleRate * duration);
    const samples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      // Frecuencias vocales formantes típicas + modulación
      const s1 = Math.sin(2 * Math.PI * 300 * t) * 0.4;
      const s2 = Math.sin(2 * Math.PI * 800 * t) * 0.3;
      const s3 = Math.sin(2 * Math.PI * 2200 * t) * 0.2;
      const envelope = Math.sin(Math.PI * (i / numSamples));
      samples[i] = (s1 + s2 + s3) * envelope;
    }

    const wavBuffer = state.audioProcessor.encodeWAV(samples, sampleRate, 1);
    return new Blob([wavBuffer], { type: 'audio/wav' });
  }

  // =========================================================================
  // 9. Envío de Petición HTTP a la API /detect
  // =========================================================================
  async function sendAudioForDetection(base64Audio, audioBlob, durationSeconds, sourceLabel) {
    showAnalyzingState('Enviando stream WAV (8kHz, Base64) a la API /detect en Canal 0...');
    state.lastAudioBlob = audioBlob;

    const payload = {
      audio_base64: base64Audio,
      channel: 0, // Parámetro exigido para simular receptor de llamada
    };

    const startTime = performance.now();

    try {
      const response = await fetch(state.apiUrl, {
        method: 'POST',
        mode: state.corsMode || 'cors',
        credentials: state.corsCredentials || 'omit',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const latencyMs = Math.round(performance.now() - startTime);

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          errorDetail = JSON.stringify(errJson.detail || errJson);
        } catch (e) { }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      state.lastResult = data;

      // Renderizar resultado
      renderDetectionResult({
        isSynthetic: Boolean(data.is_synthetic),
        confidence: Number(data.confidence),
        rawResponse: data,
        latencyMs,
        durationSeconds,
        sourceLabel,
        audioBlob,
        payloadSizeKb: Math.round((base64Audio.length * 0.75) / 1024),
      });

    } catch (err) {
      showIdleState();
      console.error('Error en /detect:', err);
      let errorMsg = err.message;
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        errorMsg = `Fallo en la comunicación (posible bloqueo CORS o conexión rechazada).\n\nDetalles:\n- URL: ${state.apiUrl}\n- Modo CORS: ${state.corsMode}\n- Credenciales: ${state.corsCredentials}\n\nAsegúrate de que la API externa tenga habilitado 'Access-Control-Allow-Origin' para admitir llamadas desde este navegador.`;
      }
      alert(`Error al comunicarse con la API de detección:\n\n${errorMsg}`);
    }
  }

  // =========================================================================
  // 10. Renderizado y Animación de Resultados
  // =========================================================================
  function renderDetectionResult(res) {
    // Ocultar estados previos y mostrar tarjeta activa
    elements.resultIdle.style.display = 'none';
    elements.resultAnalyzing.style.display = 'none';
    elements.resultActive.style.display = 'flex';

    const isSynthetic = res.isSynthetic;
    const confidencePct = (res.confidence * 100).toFixed(1);

    // 1. Configurar Tarjeta de Veredicto
    if (isSynthetic) {
      elements.verdictCard.className = 'verdict-hero-card synthetic';
      elements.verdictLabel.textContent = 'Clasificación Confirmada';
      elements.verdictTitle.textContent = 'VOZ SINTÉTICA (IA)';
      elements.verdictDesc.textContent = 'El modelo ha detectado patrones artificiales de latencia y consistencia característicos de un agente de IA.';
      elements.verdictIcon.innerHTML = `
        <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
      `;
    } else {
      elements.verdictCard.className = 'verdict-hero-card human';
      elements.verdictLabel.textContent = 'Clasificación Confirmada';
      elements.verdictTitle.textContent = 'PERSONA REAL (HUMANO)';
      elements.verdictDesc.textContent = 'El análisis de comportamiento temporal y habla corresponde a una respuesta biológica humana natural.';
      elements.verdictIcon.innerHTML = `
        <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
      `;
    }

    // 2. Animar Medidor Radial de Confianza
    // Circunferencia del círculo con r=36 es 2 * PI * 36 ≈ 226.19
    const circumference = 226.19;
    const offset = circumference - (res.confidence * circumference);
    elements.gaugeCircle.style.strokeDashoffset = offset;

    if (isSynthetic) {
      elements.gaugeCircle.style.stroke = 'var(--ai-ruby)';
      elements.progressBarFill.style.background = 'linear-gradient(90deg, #f59e0b, #f43f5e)';
    } else {
      elements.gaugeCircle.style.stroke = 'var(--human-green)';
      elements.progressBarFill.style.background = 'linear-gradient(90deg, #38bdf8, #10b981)';
    }

    elements.progressBarFill.style.width = `${confidencePct}%`;
    elements.gaugeText.textContent = `${confidencePct}%`;

    // Etiqueta de Certeza
    if (res.confidence >= 0.75) {
      elements.gaugeTierPill.textContent = 'Certeza Alta';
      elements.gaugeTierPill.style.color = '#38bdf8';
    } else if (res.confidence >= 0.58) {
      elements.gaugeTierPill.textContent = 'Certeza Moderada';
      elements.gaugeTierPill.style.color = '#fbbf24';
    } else {
      elements.gaugeTierPill.textContent = 'Certeza Baja';
      elements.gaugeTierPill.style.color = '#94a3b8';
    }

    // 3. Telemetría y Métricas
    elements.metricChannel.textContent = '0 (Receptor)';
    elements.metricRate.textContent = '8000 Hz';
    elements.metricLatency.textContent = `${res.latencyMs} ms`;
    elements.metricSize.textContent = `${res.payloadSizeKb} KB`;

    // 4. Audio Playback
    if (res.audioBlob) {
      const audioUrl = URL.createObjectURL(res.audioBlob);
      elements.audioPlayer.src = audioUrl;
    }

    // 5. Visor JSON Crudo
    elements.jsonResponsePre.textContent = JSON.stringify(res.rawResponse, null, 2);

    // 6. Agregar al Historial
    addToHistory(res);
  }

  function showAnalyzingState(subtext = 'Analizando llamada...') {
    elements.resultIdle.style.display = 'none';
    elements.resultActive.style.display = 'none';
    elements.resultAnalyzing.style.display = 'flex';
    document.querySelector('.analyzing-text-sub').textContent = subtext;
  }

  function showIdleState() {
    elements.resultAnalyzing.style.display = 'none';
    elements.resultActive.style.display = 'none';
    elements.resultIdle.style.display = 'flex';
  }

  // =========================================================================
  // 11. Copiar JSON al Portapapeles
  // =========================================================================
  elements.btnCopyJson.addEventListener('click', () => {
    if (!state.lastResult) return;
    navigator.clipboard.writeText(JSON.stringify(state.lastResult, null, 2))
      .then(() => {
        elements.btnCopyJson.textContent = '¡Copiado!';
        setTimeout(() => {
          elements.btnCopyJson.textContent = 'Copiar JSON';
        }, 2000);
      })
      .catch((err) => console.error('Error al copiar:', err));
  });

  // =========================================================================
  // 12. Historial de Auditoría de Sesión
  // =========================================================================
  function addToHistory(item) {
    state.history.unshift(item);
    if (elements.historyEmptyRow) {
      elements.historyEmptyRow.style.display = 'none';
    }

    const tr = document.createElement('tr');
    const timeStr = new Date().toLocaleTimeString();
    const confPct = (item.confidence * 100).toFixed(1);
    const badgeClass = item.isSynthetic ? 'synthetic' : 'human';
    const badgeLabel = item.isSynthetic ? 'Voz IA (Sintética)' : 'Persona Real';

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); color: var(--text-secondary);">${timeStr}</td>
      <td style="font-weight: 500;">${item.sourceLabel}</td>
      <td><span class="badge-result ${badgeClass}">${badgeLabel}</span></td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${confPct}%</td>
      <td style="font-family: var(--font-mono); color: var(--text-muted);">${item.latencyMs} ms</td>
      <td>
        <button class="btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" data-index="${state.history.length - 1}">
          Escuchar
        </button>
      </td>
    `;

    // Botón de reescuchar
    const listenBtn = tr.querySelector('button');
    listenBtn.addEventListener('click', () => {
      if (item.audioBlob) {
        elements.audioPlayer.src = URL.createObjectURL(item.audioBlob);
        elements.audioPlayer.play();
      }
    });

    elements.historyTableBody.prepend(tr);
  }

  elements.btnClearHistory.addEventListener('click', () => {
    state.history = [];
    elements.historyTableBody.innerHTML = `
      <tr id="historyEmptyRow">
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No hay registros en esta sesión todavía.
        </td>
      </tr>
    `;
    elements.historyEmptyRow = document.getElementById('historyEmptyRow');
  });

  elements.btnExportHistory.addEventListener('click', () => {
    if (state.history.length === 0) {
      alert('No hay registros para exportar');
      return;
    }

    const exportData = state.history.map((h) => ({
      timestamp: new Date().toISOString(),
      source: h.sourceLabel,
      is_synthetic: h.isSynthetic,
      confidence: h.confidence,
      latency_ms: h.latencyMs,
      duration_seconds: h.durationSeconds,
    }));

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `altur-detections-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
});
