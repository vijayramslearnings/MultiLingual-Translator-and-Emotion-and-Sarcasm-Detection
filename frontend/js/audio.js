/**
 * Bhasha Live — Real-Time Web Audio Recording & Visualizer Engine
 * Captures live microphone audio, renders spectrum visualizer, 
 * and encodes raw PCM to 16kHz 16-bit mono WAV Blob for the ML backend.
 */

(function () {
  let audioCtx = null;
  let analyser = null;
  let microphoneStream = null;
  let scriptProcessor = null;
  let isListening = false;
  let animId = null;

  // Audio recording buffer
  let recordedSamples = [];
  let recordingSampleRate = 44100;

  const canvas = document.getElementById("audioVisualizerCanvas");
  const ctx = canvas ? canvas.getContext("2d") : null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContextClass();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    recordingSampleRate = audioCtx.sampleRate;
  }

  // Draw Audio Visualizer Bars
  function drawVisualizer() {
    if (!canvas || !ctx || !analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i];
    }
    const avgAmp = sum / (bufferLength * 255);
    if (window.setParticleAudioAmplitude) {
      window.setParticleAudioAmplitude(avgAmp * 1.8);
    }

    const barWidth = (width / (bufferLength * 0.75));
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height * 0.9;

      const grad = ctx.createLinearGradient(0, height, 0, 0);
      grad.addColorStop(0, "rgba(99, 102, 241, 0.3)");
      grad.addColorStop(0.6, "rgba(139, 92, 246, 0.8)");
      grad.addColorStop(1, "rgba(6, 182, 212, 1)");

      ctx.fillStyle = grad;
      ctx.fillRect(x, height - barHeight, barWidth - 1, barHeight);

      x += barWidth;
    }

    animId = requestAnimationFrame(drawVisualizer);
  }

  // Start live microphone capture & recording
  async function startMicrophone() {
    try {
      initAudioContext();
      recordedSamples = [];

      microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        },
        video: false
      });

      const source = audioCtx.createMediaStreamSource(microphoneStream);
      source.connect(analyser);

      // Create recorder processor node with zero-gain sink to prevent echo muting
      scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessor.onaudioprocess = (e) => {
        if (!isListening) return;
        const channelData = e.inputBuffer.getChannelData(0);
        recordedSamples.push(new Float32Array(channelData));
      };

      const muteGain = audioCtx.createGain();
      muteGain.gain.value = 0.0;

      source.connect(scriptProcessor);
      scriptProcessor.connect(muteGain);
      muteGain.connect(audioCtx.destination);

      isListening = true;
      drawVisualizer();
      return true;
    } catch (err) {
      console.warn("[WebAudio] Microphone access error:", err);
      isListening = false;
      return false;
    }
  }

  // Stop recording and compile WAV Blob
  function stopMicrophone() {
    isListening = false;

    if (scriptProcessor) {
      scriptProcessor.disconnect();
      scriptProcessor = null;
    }
    if (microphoneStream) {
      microphoneStream.getTracks().forEach((track) => track.stop());
      microphoneStream = null;
    }
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    if (window.setParticleAudioAmplitude) {
      window.setParticleAudioAmplitude(0.0);
    }

    // Export 16kHz mono WAV Blob
    return exportWavBlob(recordedSamples, recordingSampleRate);
  }

  // Pure JavaScript PCM to WAV Encoder
  function exportWavBlob(chunks, sampleRate) {
    let totalLength = 0;
    for (const chunk of chunks) {
      totalLength += chunk.length;
    }

    const flatSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      flatSamples.set(chunk, offset);
      offset += chunk.length;
    }

    // Downsample to 16000 if recording was at 44.1k or 48k
    const targetSampleRate = 16000;
    let finalSamples = flatSamples;
    if (sampleRate !== targetSampleRate) {
      const ratio = sampleRate / targetSampleRate;
      const newLength = Math.round(flatSamples.length / ratio);
      finalSamples = new Float32Array(newLength);
      for (let i = 0; i < newLength; i++) {
        finalSamples[i] = flatSamples[Math.min(flatSamples.length - 1, Math.round(i * ratio))];
      }
    }

    // Convert float32 to int16 PCM
    const buffer = new ArrayBuffer(44 + finalSamples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + finalSamples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // SubChunk1Size (16 for PCM)
    view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true);  // NumChannels (1 = Mono)
    view.setUint32(24, targetSampleRate, true); // SampleRate (16000)
    view.setUint32(28, targetSampleRate * 2, true); // ByteRate
    view.setUint16(32, 2, true);  // BlockAlign
    view.setUint16(34, 16, true); // BitsPerSample (16-bit)

    // data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, finalSamples.length * 2, true);

    // Write PCM samples
    let sampleOffset = 44;
    for (let i = 0; i < finalSamples.length; i++) {
      const s = Math.max(-1, Math.min(1, finalSamples[i]));
      view.setInt16(sampleOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      sampleOffset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Simulated Audio for quick test clicks
  let simOsc = null;
  let simGain = null;

  function startSimulatedAudio() {
    initAudioContext();
    if (simOsc) {
      try { simOsc.stop(); } catch(e){}
    }

    simOsc = audioCtx.createOscillator();
    simGain = audioCtx.createGain();

    simOsc.type = "sawtooth";
    simOsc.frequency.setValueAtTime(200, audioCtx.currentTime);
    simOsc.frequency.exponentialRampToValueAtTime(320, audioCtx.currentTime + 0.8);
    simOsc.frequency.exponentialRampToValueAtTime(160, audioCtx.currentTime + 1.8);

    simGain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    simGain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 2.5);

    simOsc.connect(analyser);
    simOsc.start();
    drawVisualizer();

    setTimeout(() => {
      stopSimulatedAudio();
    }, 2600);
  }

  function stopSimulatedAudio() {
    if (simOsc) {
      try { simOsc.stop(); } catch (e) {}
      simOsc = null;
    }
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    if (window.setParticleAudioAmplitude) {
      window.setParticleAudioAmplitude(0.0);
    }
  }

  window.BhashaAudio = {
    startMicrophone,
    stopMicrophone,
    startSimulatedAudio,
    stopSimulatedAudio,
    isListening: () => isListening
  };
})();
