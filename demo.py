"""
Interactive Demo & Streamlit Web UI Dashboard for the Multilingual Voice Translation System.
Can be executed as:
  1. Streamlit Web App: streamlit run demo.py
  2. CLI Interactive Demo: python3 demo.py
"""

import sys
import os
import json
import argparse
from pipeline import MultilingualVoicePipeline
from pipeline.utils.audio_utils import generate_synthetic_audio


def run_cli_demo(audio_path: str = None, target_lang: str = "en", source_lang: str = "hi-en"):
    """Runs pipeline demo in command line interface."""
    if not audio_path or not os.path.exists(audio_path):
        print("[Demo] Generating synthetic Hinglish speech sample...")
        audio_path = "data/sample_hinglish.wav"
        generate_synthetic_audio(audio_path, duration_sec=3.0, tone_type="sarcastic")

    print(f"\n========================================================")
    print(f" Multilingual Voice Translation System — Interactive Demo")
    print(f" Processing Audio: {audio_path}")
    print(f" Source Language:  {source_lang}")
    print(f" Target Language:  {target_lang}")
    print(f"========================================================\n")

    pipeline = MultilingualVoicePipeline(config_path="config/default_config.yaml")
    result = pipeline.process(audio_path, target_lang=target_lang, source_lang=source_lang)

    print("\n--- FINAL STRUCTURED RESULT JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n========================================================\n")


def run_streamlit_dashboard():
    """Renders rich interactive Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Multilingual Voice Translation & XAI System",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Multilingual Indian Voice Translation System")
    st.markdown("""
    **Code-Mixing Support • Emotion/Sarcasm Detection • Explainable AI (XAI)**
    *A non-agentic, fully deterministic multi-stage speech & NLP pipeline.*
    """)

    st.sidebar.header("⚙️ Pipeline Configuration")
    source_lang = st.sidebar.selectbox(
        "I Speak (Source Language)",
        options=["hi-en", "ta-en", "te-en", "hi", "ta", "te", "en"],
        format_func=lambda x: {
            "hi-en": "Hinglish (Hindi + English)",
            "ta-en": "Tanglish (Tamil + English)",
            "te-en": "Telglish (Telugu + English)",
            "hi": "Hindi (हिन्दी)",
            "ta": "Tamil (தமிழ்)",
            "te": "Telugu (తెలుగు)",
            "en": "English"
        }[x]
    )

    target_lang = st.sidebar.selectbox(
        "Target Language for MT",
        options=["en", "hi", "ta", "te", "kn", "ml", "bn", "mr", "gu", "pa", "ur"],
        format_func=lambda x: {
            "en": "English (en)", "hi": "Hindi (hi)", "ta": "Tamil (ta)", 
            "te": "Telugu (te)", "kn": "Kannada (kn)", "ml": "Malayalam (ml)",
            "bn": "Bengali (bn)", "mr": "Marathi (mr)", "gu": "Gujarati (gu)",
            "pa": "Punjabi (pa)", "ur": "Urdu (ur)"
        }[x]
    )

    input_source = st.sidebar.radio(
        "Audio Input Method",
        ["Preset Benchmark Audio", "Upload Custom Audio (.wav / .mp3)"]
    )

    os.makedirs("data", exist_ok=True)
    if input_source == "Preset Benchmark Audio":
        sample_choice = st.sidebar.selectbox(
            "Select Audio Sample",
            ["Hinglish Sarcastic (Hindi-English)", "Tanglish Happy (Tamil-English)", "Telugu Neutral (Telugu-English)"]
        )
        if "Hinglish" in sample_choice:
            audio_file = "data/demo_hinglish.wav"
            generate_synthetic_audio(audio_file, duration_sec=3.0, tone_type="sarcastic")
        elif "Tanglish" in sample_choice:
            audio_file = "data/demo_tanglish.wav"
            generate_synthetic_audio(audio_file, duration_sec=3.0, tone_type="happy")
        else:
            audio_file = "data/demo_telugu.wav"
            generate_synthetic_audio(audio_file, duration_sec=3.0, tone_type="neutral")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "flac"])
        if uploaded_file is not None:
            audio_file = os.path.join("data", f"uploaded_{uploaded_file.name}")
            with open(audio_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            audio_file = "data/sample_hinglish.wav"
            if not os.path.exists(audio_file):
                generate_synthetic_audio(audio_file, duration_sec=3.0, tone_type="sarcastic")

    with st.sidebar.expander("🛠️ Advanced Settings"):
        override_text = st.text_input("Override Transcript (optional)", "")
        override_text = override_text.strip() if override_text.strip() else None

    st.subheader("🔊 Audio Signal Preview")
    st.audio(audio_file)

    if st.button("🚀 Process Voice Utterance", type="primary"):
        with st.spinner("Processing speech through deterministic 7-stage pipeline..."):
            pipeline = MultilingualVoicePipeline(config_path="config/default_config.yaml")
            res = pipeline.process(
                audio_file, 
                target_lang=target_lang, 
                source_lang=source_lang,
                override_transcript=override_text
            )

        st.success("Utterance Processed Successfully!")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📝 Transcript & Language Segmentation")
            st.info(f"**Raw Transcript:** {res['transcript']}")
            
            st.markdown("**Language Segments (LID):**")
            seg_html = ""
            for seg in res["language_segments"]:
                color = "#4A90E2" if seg["lang"] == "en" else "#E67E22"
                seg_html += f'<span style="background-color:{color}; color:white; padding:4px 8px; margin:2px; border-radius:4px; font-weight:bold;">{seg["text"]} ({seg["lang"]})</span> '
            st.markdown(seg_html, unsafe_allow_html=True)

            st.markdown("### 🌐 Machine Translation")
            st.success(f"**Target ({res['target_language']}):** {res['translated_text']}")

        with col2:
            st.markdown("### 🎭 Emotion Detection & Modality Fusion")
            emo = res["emotion"]
            st.metric("Final Emotion Label", f"{emo['label'].upper()}", f"Confidence: {emo['confidence']*100:.1f}%")
            
            st.write(f"**Text Modality:** `{emo['text_modality']['label']}` ({emo['text_modality']['confidence']*100:.1f}%)")
            st.write(f"**Acoustic Modality:** `{emo['acoustic_modality']['label']}` ({emo['acoustic_modality']['confidence']*100:.1f}%)")
            st.write(f"**Modality Agreement:** {'✅ Yes' if emo['modality_agreement'] else '❌ Disagreement (Tone Mismatch)'}")

            st.markdown("### 😼 Sarcasm Detection")
            sarcasm = res["sarcasm"]
            if sarcasm["present"]:
                st.warning(f"**Sarcasm Present!** (Incongruity Score: {sarcasm['incongruity_score']:.2f})")
            else:
                st.info(f"**Literal Utterance** (Incongruity Score: {sarcasm['incongruity_score']:.2f})")
            st.caption(f"**Reasoning:** {sarcasm['reason']}")

        st.markdown("---")
        st.markdown("### 🧠 Explainable AI (XAI) Attributions")
        xai_col1, xai_col2 = st.columns([1, 1])

        with xai_col1:
            st.markdown("**Token-Level Integrated Gradients (Text Attribution):**")
            for attr in res["explanation"]["top_attributed_tokens"]:
                st.progress(min(1.0, float(attr["score"] * 2)), text=f"Token: '{attr['token']}' — Score: {attr['score']:.4f}")

        with xai_col2:
            st.markdown("**Acoustic Time-Frequency Spectrogram Cues:**")
            for cue in res["explanation"]["acoustic_cues"]:
                st.markdown(f"• {cue}")

        st.markdown("---")
        st.markdown("### 📄 Structured Result JSON")
        st.json(res)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = (get_script_run_ctx() is not None)
    except Exception:
        in_streamlit = False

    if in_streamlit:
        run_streamlit_dashboard()
    else:
        parser = argparse.ArgumentParser(description="Multilingual Voice Translation Demo")
        parser.add_argument("--audio", type=str, default=None, help="Path to input audio file")
        parser.add_argument("--target_lang", type=str, default="en", help="Target language code (e.g. en, hi, ta)")
        parser.add_argument("--web", action="store_true", help="Launch Streamlit web UI")
        args = parser.parse_args()

        if args.web:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        else:
            run_cli_demo(args.audio, args.target_lang)
