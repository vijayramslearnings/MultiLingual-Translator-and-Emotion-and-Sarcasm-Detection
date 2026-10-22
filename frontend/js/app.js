/**
 * Bhasha Live — Clean Two-Way Voice & Text Translation Controller
 * Supports:
 *  - Prominent Language Dropdowns for Speaker A and Speaker B
 *  - Both Real-Time Voice Recording (Microphone) and Instant Typed Text
 *  - Real-Time Emotion, Sarcasm, and Confidence Detection Rendering
 *  - Speech Synthesis (TTS) Playback
 *  - One-click Text Copy with Toast Notification
 *  - Speaker/Language Role Swapping
 *  - Interactive Avatar Cycling
 *  - Light / Dark Theme Switching
 */

(function () {
  "use strict";

  // Application State
  const state = {
    theme: localStorage.getItem("bhasha_theme") || "dark",
    isRecording: false,
    activeSpeaker: null, // "A" or "B"
    ttsSpeed: 1.0,
    avatarIndexA: 0,
    avatarIndexB: 1
  };

  // API Base Resolution (supports both direct server hosting and file:/// preview)
  let apiBaseUrl = "";
  if (window.location.protocol === "file:") {
    apiBaseUrl = localStorage.getItem("bhasha_api_base") || "http://localhost:8000";
  }

  function getApiUrl(path) {
    if (!path.startsWith("/")) path = "/" + path;
    if (window.location.protocol.startsWith("http")) {
      return path;
    }
    return `${apiBaseUrl}${path}`;
  }

  // Automatic backend connectivity discovery for file:/// mode
  async function checkBackendConnectivity() {
    if (window.location.protocol !== "file:") return;
    const candidates = [
      "http://localhost:8000",
      "http://127.0.0.1:8000",
      "http://localhost:3000",
      "http://127.0.0.1:3000",
      "https://localhost:8000",
      "https://localhost:3000"
    ];
    for (const host of candidates) {
      try {
        const res = await fetch(`${host}/api/health`, { method: "GET", mode: "cors" });
        if (res.ok) {
          apiBaseUrl = host;
          localStorage.setItem("bhasha_api_base", host);
          console.log(`[Bhasha] Connected to backend at ${host}`);
          return;
        }
      } catch (e) {
        // try next candidate
      }
    }
  }
  checkBackendConnectivity();

  // Avatar assets from user's design
  const avatarList = [
    "assets/avatar_man_blue.png",
    "assets/avatar_woman_red.png",
    "assets/avatar_man_green.png",
    "assets/avatar_woman_yellow.png"
  ];

  // =========================================================================
  // 1. DOM ELEMENTS
  // =========================================================================
  // Theme
  const btnThemeLight = document.getElementById("btnThemeLight");
  const btnThemeDark = document.getElementById("btnThemeDark");
  const themeIcon = document.getElementById("themeIcon");

  // Speaker A
  const cardPersonA = document.getElementById("cardPersonA");
  const avatarRingA = document.getElementById("avatarRingA");
  const avatarImgA = document.getElementById("avatarImgA");
  const personALang = document.getElementById("personALang");
  const statusPersonA = document.getElementById("statusPersonA");
  const btnMicA = document.getElementById("btnMicA");
  const soundwaveLeftA = document.getElementById("soundwaveLeftA");
  const soundwaveRightA = document.getElementById("soundwaveRightA");
  const micPromptA = document.getElementById("micPromptA");
  const textOrigA = document.getElementById("textOrigA");
  const btnClearA = document.getElementById("btnClearA");
  const btnTranslateA = document.getElementById("btnTranslateA");
  const textTransA = document.getElementById("textTransA");
  const btnCopyTransA = document.getElementById("btnCopyTransA");
  const btnPlayTransA = document.getElementById("btnPlayTransA");

  // Speaker A Metrics
  const cardEmoA = document.getElementById("cardEmoA");
  const valEmoA = document.getElementById("valEmoA");
  const pctEmoA = document.getElementById("pctEmoA");
  const cardSarcA = document.getElementById("cardSarcA");
  const valSarcA = document.getElementById("valSarcA");
  const pctSarcA = document.getElementById("pctSarcA");
  const cardConfA = document.getElementById("cardConfA");
  const valConfA = document.getElementById("valConfA");
  const fillConfA = document.getElementById("fillConfA");
  const xaiBoxA = document.getElementById("xaiBoxA");
  const xaiEmoTextA = document.getElementById("xaiEmoTextA");
  const xaiSarcTextA = document.getElementById("xaiSarcTextA");

  // Speaker B
  const cardPersonB = document.getElementById("cardPersonB");
  const avatarRingB = document.getElementById("avatarRingB");
  const avatarImgB = document.getElementById("avatarImgB");
  const personBLang = document.getElementById("personBLang");
  const statusPersonB = document.getElementById("statusPersonB");
  const btnMicB = document.getElementById("btnMicB");
  const soundwaveLeftB = document.getElementById("soundwaveLeftB");
  const soundwaveRightB = document.getElementById("soundwaveRightB");
  const micPromptB = document.getElementById("micPromptB");
  const textOrigB = document.getElementById("textOrigB");
  const btnClearB = document.getElementById("btnClearB");
  const btnTranslateB = document.getElementById("btnTranslateB");
  const textTransB = document.getElementById("textTransB");
  const btnCopyTransB = document.getElementById("btnCopyTransB");
  const btnPlayTransB = document.getElementById("btnPlayTransB");

  // Speaker B Metrics
  const cardEmoB = document.getElementById("cardEmoB");
  const valEmoB = document.getElementById("valEmoB");
  const pctEmoB = document.getElementById("pctEmoB");
  const cardSarcB = document.getElementById("cardSarcB");
  const valSarcB = document.getElementById("valSarcB");
  const pctSarcB = document.getElementById("pctSarcB");
  const cardConfB = document.getElementById("cardConfB");
  const valConfB = document.getElementById("valConfB");
  const fillConfB = document.getElementById("fillConfB");
  const xaiBoxB = document.getElementById("xaiBoxB");
  const xaiEmoTextB = document.getElementById("xaiEmoTextB");
  const xaiSarcTextB = document.getElementById("xaiSarcTextB");

  // Common Controls
  const btnSwapSpeakers = document.getElementById("btnSwapSpeakers");
  const toastMessage = document.getElementById("toastMessage");

  // =========================================================================
  // 2. THEME SYSTEM
  // =========================================================================
  function setTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("bhasha_theme", theme);

    if (btnThemeLight && btnThemeDark) {
      if (theme === "light") {
        btnThemeLight.classList.add("active");
        btnThemeDark.classList.remove("active");
      } else {
        btnThemeDark.classList.add("active");
        btnThemeLight.classList.remove("active");
      }
    }
  }

  if (btnThemeLight) btnThemeLight.addEventListener("click", () => setTheme("light"));
  if (btnThemeDark) btnThemeDark.addEventListener("click", () => setTheme("dark"));
  setTheme(state.theme);

  // =========================================================================
  // 3. TOAST NOTIFICATION
  // =========================================================================
  let toastTimer = null;
  function showToast(msg) {
    if (!toastMessage) return;
    toastMessage.textContent = msg;
    toastMessage.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastMessage.classList.remove("show");
    }, 2500);
  }

  // =========================================================================
  // 4. LANGUAGE HELPER FUNCTIONS
  // =========================================================================
  function getLangDisplay(code) {
    const map = {
      "hi-en": "Hinglish",
      "hi": "Hindi",
      "ta-en": "Tanglish",
      "ta": "Tamil",
      "en": "English"
    };
    return map[code] || code.toUpperCase();
  }

  function getSpeakerLanguages(speaker) {
    if (speaker === "A") {
      return {
        sourceLang: personALang ? personALang.value : "hi-en",
        targetLang: personBLang ? personBLang.value : "en"
      };
    } else {
      return {
        sourceLang: personBLang ? personBLang.value : "en",
        targetLang: personALang ? personALang.value : "hi-en"
      };
    }
  }

  if (personALang) {
    personALang.addEventListener("change", () => {
      showToast(`Speaker A language set to ${getLangDisplay(personALang.value)}`);
    });
  }

  if (personBLang) {
    personBLang.addEventListener("change", () => {
      showToast(`Speaker B language set to ${getLangDisplay(personBLang.value)}`);
    });
  }

  // =========================================================================
  // 5. TEXT TRANSLATION WORKFLOW
  // =========================================================================
  async function processText(speaker) {
    const isA = speaker === "A";
    const textarea = isA ? textOrigA : textOrigB;
    const cleanText = (textarea.value || "").trim();

    if (!cleanText) {
      textarea.focus();
      showToast("Please enter some text to translate");
      return;
    }

    const { sourceLang, targetLang } = getSpeakerLanguages(speaker);
    const statusElem = isA ? statusPersonA : statusPersonB;
    const translateBtn = isA ? btnTranslateA : btnTranslateB;

    // Loading UI state
    statusElem.className = "person-status-badge status-listening";
    statusElem.querySelector(".status-text").innerText = "Translating...";
    if (translateBtn) {
      translateBtn.disabled = true;
      translateBtn.innerHTML = `
        <svg class="spin-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
        Translating...
      `;
    }

    try {
      const resp = await fetch(getApiUrl("/api/process-text"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: cleanText,
          source_lang: sourceLang,
          target_lang: targetLang
        })
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const data = await resp.json();
      applyMLResults(speaker, data, sourceLang, targetLang);
      showToast(`Translated to ${getLangDisplay(targetLang)}!`);

    } catch (err) {
      console.error("Text translation failed:", err);
      showToast("Translation error. Please check server status.");
    } finally {
      statusElem.className = "person-status-badge status-ready";
      statusElem.querySelector(".status-text").innerText = "Ready";
      if (translateBtn) {
        translateBtn.disabled = false;
        translateBtn.innerHTML = `
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/></svg>
          Translate
        `;
      }
    }
  }

  // Hook translate buttons
  if (btnTranslateA) btnTranslateA.addEventListener("click", () => processText("A"));
  if (btnTranslateB) btnTranslateB.addEventListener("click", () => processText("B"));

  // Hook Enter key (without Shift) in textareas
  if (textOrigA) {
    textOrigA.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        processText("A");
      }
    });
  }

  if (textOrigB) {
    textOrigB.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        processText("B");
      }
    });
  }

  // Clear buttons
  if (btnClearA) {
    btnClearA.addEventListener("click", () => {
      textOrigA.value = "";
      textTransA.innerText = "Translation will appear here...";
      resetMetrics("A");
      textOrigA.focus();
    });
  }

  if (btnClearB) {
    btnClearB.addEventListener("click", () => {
      textOrigB.value = "";
      textTransB.innerText = "Translation will appear here...";
      resetMetrics("B");
      textOrigB.focus();
    });
  }

  // =========================================================================
  // 6. VOICE RECORDING WORKFLOW
  // =========================================================================
  let activeSpeechRec = null;
  let activeLiveTranscript = "";

  function getSpeechRecLang(langCode) {
    const langMap = {
      "ta": "ta-IN",
      "hi": "hi-IN",
      "hi-en": "hi-IN",
      "ta-en": "ta-IN",
      "te": "te-IN",
      "te-en": "te-IN",
      "kn": "kn-IN",
      "kn-en": "kn-IN",
      "ml": "ml-IN",
      "ml-en": "ml-IN",
      "bn": "bn-IN",
      "bn-en": "bn-IN",
      "mr": "mr-IN",
      "mr-en": "mr-IN",
      "gu": "gu-IN",
      "gu-en": "gu-IN",
      "pa": "pa-IN",
      "pa-en": "pa-IN",
      "ur": "ur-PK",
      "ur-en": "ur-PK",
      "en": "en-IN",
      "es": "es-ES",
      "fr": "fr-FR",
      "de": "de-DE",
      "zh": "zh-CN",
      "ja": "ja-JP",
      "ko": "ko-KR",
      "ru": "ru-RU",
      "ar": "ar-SA"
    };
    return langMap[langCode] || (langCode ? langCode.split("-")[0] : "en-IN");
  }

  async function toggleRecording(speaker) {
    if (!window.BhashaAudio) {
      alert("Audio recorder module not loaded. Please refresh.");
      return;
    }

    if (state.isRecording) {
      if (state.activeSpeaker === speaker) {
        await stopRecording();
      } else {
        await stopRecording();
        await startRecording(speaker);
      }
    } else {
      await startRecording(speaker);
    }
  }

  async function startRecording(speaker) {
    state.activeSpeaker = speaker;
    state.isRecording = true;
    activeLiveTranscript = "";

    const targetInput = speaker === "A" ? textOrigA : textOrigB;
    if (targetInput) {
      targetInput.value = "";
      targetInput.placeholder = "Listening... Speak now...";
    }

    try {
      await window.BhashaAudio.startMicrophone();

      // Start Browser Speech Recognition in parallel for real-time live typing
      const SpeechRecClass = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecClass) {
        try {
          activeSpeechRec = new SpeechRecClass();
          activeSpeechRec.continuous = true;
          activeSpeechRec.interimResults = true;
          const { sourceLang } = getSpeakerLanguages(speaker);
          activeSpeechRec.lang = getSpeechRecLang(sourceLang);

          const targetInput = speaker === "A" ? textOrigA : textOrigB;

          activeSpeechRec.onresult = (event) => {
            let transcript = "";
            for (let i = 0; i < event.results.length; ++i) {
              transcript += event.results[i][0].transcript;
            }
            if (transcript.trim()) {
              activeLiveTranscript = transcript.trim();
              targetInput.value = activeLiveTranscript;
            }
          };

          activeSpeechRec.onerror = (e) => {
            console.warn("[SpeechRecognition] event warning:", e.error);
          };

          activeSpeechRec.start();
        } catch (recErr) {
          console.warn("[SpeechRecognition] Init warning:", recErr);
          activeSpeechRec = null;
        }
      }

      if (speaker === "A") {
        btnMicA.classList.add("recording");
        if (soundwaveLeftA) soundwaveLeftA.classList.add("active");
        if (soundwaveRightA) soundwaveRightA.classList.add("active");
        statusPersonA.className = "person-status-badge status-listening";
        statusPersonA.querySelector(".status-text").innerText = "Listening...";
        micPromptA.innerText = "Listening... Speak now, click mic to stop & translate";
      } else {
        btnMicB.classList.add("recording");
        if (soundwaveLeftB) soundwaveLeftB.classList.add("active");
        if (soundwaveRightB) soundwaveRightB.classList.add("active");
        statusPersonB.className = "person-status-badge status-listening";
        statusPersonB.querySelector(".status-text").innerText = "Listening...";
        micPromptB.innerText = "Listening... Speak now, click mic to stop & translate";
      }
    } catch (err) {
      console.error("Microphone access error:", err);
      showToast("Microphone permission required! Please allow mic access in your browser.", 5000);
      alert("Microphone permission denied or device not found. Please click 'Allow' on your browser's microphone prompt.");
      resetSpeakerUI();
    }
  }

  async function stopRecording() {
    const speaker = state.activeSpeaker || "A";
    state.isRecording = false;

    // Stop live browser speech recognition
    if (activeSpeechRec) {
      try {
        activeSpeechRec.stop();
      } catch (e) {}
      activeSpeechRec = null;
    }

    // Reset button animations
    btnMicA.classList.remove("recording");
    btnMicB.classList.remove("recording");
    if (soundwaveLeftA) soundwaveLeftA.classList.remove("active");
    if (soundwaveRightA) soundwaveRightA.classList.remove("active");
    if (soundwaveLeftB) soundwaveLeftB.classList.remove("active");
    if (soundwaveRightB) soundwaveRightB.classList.remove("active");

    const statusElem = speaker === "A" ? statusPersonA : statusPersonB;
    statusElem.className = "person-status-badge status-listening";
    statusElem.querySelector(".status-text").innerText = "Translating...";

    const promptElem = speaker === "A" ? micPromptA : micPromptB;
    promptElem.innerText = "Processing audio through ML pipeline...";

    try {
      const wavBlob = await window.BhashaAudio.stopMicrophone();
      const { sourceLang, targetLang } = getSpeakerLanguages(speaker);
      const targetInput = speaker === "A" ? textOrigA : textOrigB;
      const liveText = (activeLiveTranscript && activeLiveTranscript.trim()) ? activeLiveTranscript.trim() : "";

      const formData = new FormData();
      formData.append("audio", wavBlob, `mic_${Date.now()}.wav`);
      formData.append("source_lang", sourceLang);
      formData.append("target_lang", targetLang);
      if (liveText) {
        formData.append("live_transcript", liveText);
      }

      const resp = await fetch(getApiUrl("/api/process"), {
        method: "POST",
        body: formData
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      applyMLResults(speaker, data, sourceLang, targetLang);
      showToast(`Voice translated to ${getLangDisplay(targetLang)}!`);

    } catch (err) {
      console.error("Audio processing failed:", err);
      promptElem.innerText = "Translation error. Please check server.";
      showToast("Audio processing failed. Check microphone or server.");
    } finally {
      resetSpeakerUI();
    }
  }

  function resetSpeakerUI() {
    state.isRecording = false;
    state.activeSpeaker = null;

    if (activeSpeechRec) {
      try {
        activeSpeechRec.stop();
      } catch (e) {}
      activeSpeechRec = null;
    }

    btnMicA.classList.remove("recording");
    btnMicB.classList.remove("recording");
    if (soundwaveLeftA) soundwaveLeftA.classList.remove("active");
    if (soundwaveRightA) soundwaveRightA.classList.remove("active");
    if (soundwaveLeftB) soundwaveLeftB.classList.remove("active");
    if (soundwaveRightB) soundwaveRightB.classList.remove("active");

    statusPersonA.className = "person-status-badge status-ready";
    statusPersonA.querySelector(".status-text").innerText = "Ready";
    micPromptA.innerText = "Tap mic to speak or type text below";

    statusPersonB.className = "person-status-badge status-ready";
    statusPersonB.querySelector(".status-text").innerText = "Ready";
    micPromptB.innerText = "Tap mic to speak or type text below";
  }

  if (btnMicA) btnMicA.addEventListener("click", () => toggleRecording("A"));
  if (btnMicB) btnMicB.addEventListener("click", () => toggleRecording("B"));

  // =========================================================================
  // 7. RENDER ML RESULTS INTO CARDS & METRICS
  // =========================================================================
  function applyMLResults(speaker, data, sourceLang, targetLang) {
    const isA = speaker === "A";

    // 1. Text & Translation
    const origText = data.transcript || (isA ? textOrigA.value : textOrigB.value);
    const transText = data.translated_text || "";

    if (isA) {
      textOrigA.value = origText;
      textTransA.innerText = transText;
    } else {
      textOrigB.value = origText;
      textTransB.innerText = transText;
    }

    // 2. Emotion Metric
    const emotion = data.emotion || {};
    const emoLabel = capitalize(emotion.label || "Neutral");
    const emoConf = Math.round((emotion.confidence || 0.70) * 100);

    // 3. Sarcasm Metric
    const sarcasm = data.sarcasm || {};
    const isSarcastic = Boolean(sarcasm.present);
    const sarcScore = Math.round((sarcasm.incongruity_score || sarcasm.confidence || 0.05) * 100);

    // 4. Overall Confidence
    const overallConf = Math.round((emotion.confidence ? emotion.confidence * 0.94 : 0.92) * 100);

    // 5. XAI Explanations
    const explanation = data.explanation || {};
    const emoJust = explanation.emotion_justification || "";
    const sarcJust = explanation.sarcasm_justification || "";

    if (isA) {
      valEmoA.innerText = emoLabel;
      pctEmoA.innerText = `(${emoConf}%)`;
      valSarcA.innerText = isSarcastic ? "Detected ⚠️" : "Not Detected";
      pctSarcA.innerText = `(${sarcScore}%)`;
      valConfA.innerText = `${overallConf}%`;
      fillConfA.style.width = `${overallConf}%`;

      if (xaiBoxA && (emoJust || sarcJust)) {
        if (xaiEmoTextA) xaiEmoTextA.innerText = emoJust;
        if (xaiSarcTextA) xaiSarcTextA.innerText = sarcJust;
        xaiBoxA.style.display = "flex";
      }
    } else {
      valEmoB.innerText = emoLabel;
      pctEmoB.innerText = `(${emoConf}%)`;
      valSarcB.innerText = isSarcastic ? "Detected ⚠️" : "Not Detected";
      pctSarcB.innerText = `(${sarcScore}%)`;
      valConfB.innerText = `${overallConf}%`;
      fillConfB.style.width = `${overallConf}%`;

      if (xaiBoxB && (emoJust || sarcJust)) {
        if (xaiEmoTextB) xaiEmoTextB.innerText = emoJust;
        if (xaiSarcTextB) xaiSarcTextB.innerText = sarcJust;
        xaiBoxB.style.display = "flex";
      }
    }
  }

  function resetMetrics(speaker) {
    if (speaker === "A") {
      valEmoA.innerText = "Neutral";
      pctEmoA.innerText = "(50%)";
      valSarcA.innerText = "Not Detected";
      pctSarcA.innerText = "(0%)";
      valConfA.innerText = "90%";
      fillConfA.style.width = "90%";
      if (xaiBoxA) xaiBoxA.style.display = "none";
    } else {
      valEmoB.innerText = "Neutral";
      pctEmoB.innerText = "(50%)";
      valSarcB.innerText = "Not Detected";
      pctSarcB.innerText = "(0%)";
      valConfB.innerText = "90%";
      fillConfB.style.width = "90%";
      if (xaiBoxB) xaiBoxB.style.display = "none";
    }
  }

  function capitalize(str) {
    if (!str) return "Neutral";
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  // =========================================================================
  // 8. TEXT-TO-SPEECH (TTS) PLAYBACK
  // =========================================================================
  function speakText(text, langCode) {
    if (!window.speechSynthesis) {
      alert("Text-to-speech is not supported on this browser.");
      return;
    }
    const clean = (text || "").trim();
    if (!clean || clean.startsWith("Translation will appear")) {
      showToast("No translated text to play");
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = state.ttsSpeed || 1.0;
    utterance.pitch = 1.0;

    // Pick appropriate BCP-47 tag
    const l = (langCode || "").toLowerCase();
    if (l.includes("ta")) {
      utterance.lang = "ta-IN";
    } else if (l.includes("hi")) {
      utterance.lang = "hi-IN";
    } else {
      utterance.lang = "en-IN";
    }

    window.speechSynthesis.speak(utterance);
    showToast(`Playing audio (${getLangDisplay(langCode)})...`);
  }

  if (btnPlayTransA) {
    btnPlayTransA.addEventListener("click", () => {
      const { targetLang } = getSpeakerLanguages("A");
      speakText(textTransA.innerText, targetLang);
    });
  }

  if (btnPlayTransB) {
    btnPlayTransB.addEventListener("click", () => {
      const { targetLang } = getSpeakerLanguages("B");
      speakText(textTransB.innerText, targetLang);
    });
  }

  // =========================================================================
  // 9. COPY TRANSLATION TO CLIPBOARD
  // =========================================================================
  function copyToClipboard(text) {
    const clean = (text || "").trim();
    if (!clean || clean.startsWith("Translation will appear")) {
      showToast("Nothing to copy");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(clean).then(() => {
        showToast("Translation copied to clipboard!");
      }).catch(() => {
        fallbackCopy(clean);
      });
    } else {
      fallbackCopy(clean);
    }
  }

  function fallbackCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    showToast("Translation copied to clipboard!");
  }

  if (btnCopyTransA) {
    btnCopyTransA.addEventListener("click", () => copyToClipboard(textTransA.innerText));
  }

  if (btnCopyTransB) {
    btnCopyTransB.addEventListener("click", () => copyToClipboard(textTransB.innerText));
  }

  // =========================================================================
  // 10. ROLE & LANGUAGE SWAP BUTTON
  // =========================================================================
  function swapSpeakers() {
    // 1. Swap dropdown selections
    const tempLang = personALang.value;
    personALang.value = personBLang.value;
    personBLang.value = tempLang;

    // 2. Swap text inputs
    const tempOrig = textOrigA.value;
    textOrigA.value = textOrigB.value;
    textOrigB.value = tempOrig;

    // 3. Swap translations
    const tempTrans = textTransA.innerText;
    textTransA.innerText = textTransB.innerText;
    textTransB.innerText = tempTrans;

    // 4. Swap metrics
    const tempEmo = valEmoA.innerText;
    valEmoA.innerText = valEmoB.innerText;
    valEmoB.innerText = tempEmo;

    const tempSarc = valSarcA.innerText;
    valSarcA.innerText = valSarcB.innerText;
    valSarcB.innerText = tempSarc;

    // Visual button rotation feedback
    if (btnSwapSpeakers) {
      btnSwapSpeakers.style.transform = "rotate(360deg) scale(1.15)";
      setTimeout(() => {
        btnSwapSpeakers.style.transform = "";
      }, 400);
    }

    showToast("Languages and text swapped!");
  }

  if (btnSwapSpeakers) {
    btnSwapSpeakers.addEventListener("click", swapSpeakers);
  }

  // =========================================================================
  // 11. AVATAR INTERACTION (Cycle through avatars on click)
  // =========================================================================
  if (avatarRingA) {
    avatarRingA.addEventListener("click", () => {
      state.avatarIndexA = (state.avatarIndexA + 1) % avatarList.length;
      if (avatarImgA) avatarImgA.src = avatarList[state.avatarIndexA];
      avatarRingA.style.transform = "scale(1.15)";
      setTimeout(() => { avatarRingA.style.transform = ""; }, 200);
    });
  }

  if (avatarRingB) {
    avatarRingB.addEventListener("click", () => {
      state.avatarIndexB = (state.avatarIndexB + 1) % avatarList.length;
      if (avatarImgB) avatarImgB.src = avatarList[state.avatarIndexB];
      avatarRingB.style.transform = "scale(1.15)";
      setTimeout(() => { avatarRingB.style.transform = ""; }, 200);
    });
  }

})();
