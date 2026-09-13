/**
 * visualizer.js
 * Visualizador de audio reactivo en tiempo real sobre Canvas con aceleración gráfica.
 * Modos: Espectro de Frecuencia con barras luminosas, Osciloscopio de Onda y Estado Ambiente.
 */

class AudioVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      console.warn(`Canvas con id ${canvasId} no encontrado`);
      return;
    }
    this.ctx = this.canvas.getContext('2d');
    this.analyser = null;
    this.animationId = null;
    this.mode = 'combined'; // 'wave', 'bars', 'combined'
    this.isActive = false;
    this.idlePhase = 0;
    
    // Configurar resolución retina
    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Iniciar loop de renderizado ambiente
    this.drawIdle();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
  }

  /**
   * Conecta el analizador de Web Audio API
   * @param {AnalyserNode} analyserNode 
   */
  connect(analyserNode) {
    this.analyser = analyserNode;
    this.isActive = true;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.render();
  }

  /**
   * Desconecta el analizador y vuelve al estado idle suave
   */
  disconnect() {
    this.isActive = false;
    this.analyser = null;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.drawIdle();
  }

  render() {
    if (!this.isActive || !this.analyser) {
      this.drawIdle();
      return;
    }

    this.animationId = requestAnimationFrame(() => this.render());

    const bufferLength = this.analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);
    const timeData = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(freqData);
    this.analyser.getByteTimeDomainData(timeData);

    const w = this.width;
    const h = this.height;

    // Limpiar canvas con fade sutil para efecto estela
    this.ctx.clearRect(0, 0, w, h);

    // Dibujar grid de fondo sutil estilo Altur
    this.drawGrid(w, h);

    // 1. Dibujar Barras de Espectro de Fondo (Frecuencias)
    const barCount = 48;
    const barWidth = (w / barCount) - 3;
    const step = Math.floor(bufferLength / barCount);

    for (let i = 0; i < barCount; i++) {
      const value = freqData[i * step] / 255;
      const barHeight = Math.max(4, value * (h * 0.75));
      const x = i * (barWidth + 3) + 1.5;
      const y = h - barHeight - 8;

      // Gradiente Altur (Cian -> Azul Eléctrico -> Violeta)
      const grad = this.ctx.createLinearGradient(0, y, 0, h);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.85)'); // Cyan luminoso
      grad.addColorStop(0.5, 'rgba(99, 102, 241, 0.6)'); // Indigo
      grad.addColorStop(1, 'rgba(139, 92, 246, 0.15)'); // Violeta suave

      this.ctx.fillStyle = grad;
      this.ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
      this.ctx.shadowBlur = 8;
      
      // Barra con esquinas redondeadas
      this.drawRoundedRect(x, y, barWidth, barHeight, 3);
      this.ctx.fill();
    }
    this.ctx.shadowBlur = 0;

    // 2. Dibujar Forma de Onda (Osciloscopio suave al frente)
    this.ctx.lineWidth = 2.5;
    const waveGrad = this.ctx.createLinearGradient(0, 0, w, 0);
    waveGrad.addColorStop(0, 'rgba(52, 211, 153, 0.9)'); // Verde esmeralda (Voz activa)
    waveGrad.addColorStop(0.5, 'rgba(56, 189, 248, 1)'); // Cian brillante
    waveGrad.addColorStop(1, 'rgba(168, 85, 247, 0.9)'); // Púrpura

    this.ctx.strokeStyle = waveGrad;
    this.ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
    this.ctx.shadowBlur = 12;

    this.ctx.beginPath();
    const sliceWidth = w / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = timeData[i] / 128.0; // 0.0 a 2.0
      const y = (v * h) / 2;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    this.ctx.stroke();
    this.ctx.shadowBlur = 0;
  }

  drawIdle() {
    if (this.isActive) return;

    this.idlePhase += 0.03;
    const w = this.width;
    const h = this.height;

    this.ctx.clearRect(0, 0, w, h);
    this.drawGrid(w, h);

    // Dibujar onda sinusoidal suave en estado reposo
    this.ctx.lineWidth = 1.5;
    this.ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    this.ctx.beginPath();

    const midY = h / 2;
    for (let x = 0; x < w; x += 2) {
      const wave1 = Math.sin((x * 0.015) + this.idlePhase) * 12;
      const wave2 = Math.cos((x * 0.03) - (this.idlePhase * 0.8)) * 6;
      const y = midY + wave1 + wave2;

      if (x === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }
    this.ctx.stroke();

    // Dibujar barras idle sutiles
    const barCount = 48;
    const barWidth = (w / barCount) - 3;
    for (let i = 0; i < barCount; i++) {
      const val = (Math.sin((i * 0.2) + this.idlePhase) + 1) * 0.5;
      const barH = 4 + val * 10;
      const x = i * (barWidth + 3) + 1.5;
      const y = h - barH - 8;

      this.ctx.fillStyle = 'rgba(255, 255, 255, 0.06)';
      this.drawRoundedRect(x, y, barWidth, barH, 2);
      this.ctx.fill();
    }

    this.animationId = requestAnimationFrame(() => this.drawIdle());
  }

  drawGrid(w, h) {
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    this.ctx.lineWidth = 1;

    // Línea central
    this.ctx.beginPath();
    this.ctx.moveTo(0, h / 2);
    this.ctx.lineTo(w, h / 2);
    this.ctx.stroke();

    // Líneas verticales tenues
    for (let x = 0; x < w; x += 40) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
      this.ctx.stroke();
    }
  }

  drawRoundedRect(x, y, width, height, radius) {
    this.ctx.beginPath();
    this.ctx.moveTo(x + radius, y);
    this.ctx.lineTo(x + width - radius, y);
    this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    this.ctx.lineTo(x + width, y + height - radius);
    this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    this.ctx.lineTo(x + radius, y + height);
    this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    this.ctx.lineTo(x, y + radius);
    this.ctx.quadraticCurveTo(x, y, x + radius, y);
    this.ctx.closePath();
  }
}

// Exportar como objeto global
window.AudioVisualizer = AudioVisualizer;
