#!/usr/bin/env python3
"""
Generates an ultra-crisp, publication-grade Architecture Flowchart Diagram (PNG)
for the Multilingual Indian Voice Translation, Emotion/Sarcasm Detection & XAI System.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_rounded_box(ax, x, y, w, h, title, subtitle="", badge="", color="#1E293B", border_color="#38BDF8", text_color="#FFFFFF"):
    # Draw rounded rectangle
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.8,
        zorder=3
    )
    ax.add_patch(box)

    # Text positioning
    if badge:
        ax.text(x + 0.04, y + h - 0.08, badge.upper(), fontsize=7.5, fontweight='bold', color=border_color, zorder=4)
        ax.text(x + 0.04, y + h - 0.22, title, fontsize=10.5, fontweight='bold', color=text_color, zorder=4)
    else:
        ax.text(x + 0.04, y + h - 0.16, title, fontsize=11, fontweight='bold', color=text_color, zorder=4)

    if subtitle:
        ax.text(x + 0.04, y + 0.08, subtitle, fontsize=8.2, color="#94A3B8", style='italic', zorder=4)

def draw_arrow(ax, start, end, label="", color="#38BDF8", style="-|>", rad=0.0, lw=1.8):
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=connectionstyle
        ),
        zorder=2
    )
    if label:
        mx = (start[0] + end[0]) / 2 + (0.05 if rad == 0 else 0.08)
        my = (start[1] + end[1]) / 2
        ax.text(mx, my, label, fontsize=8, color="#F8FAFC", fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="none", alpha=0.85),
                zorder=5, ha='center', va='center')

def create_architecture_diagram(output_path="architecture_flowchart.png"):
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor('#0F172A') # Dark slate background
    ax.set_facecolor('#0F172A')
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, 11.5)
    ax.axis('off')

    # Header Title
    ax.text(8.5, 11.0, "BHASHA: END-TO-END MULTILINGUAL VOICE TRANSLATION & XAI SYSTEM ARCHITECTURE",
            fontsize=16, fontweight='bold', color='#FFFFFF', ha='center', va='center')
    ax.text(8.5, 10.65, "Deterministic 7-Stage Pipeline with Code-Mixing Normalization, Multimodal Emotion Fusion & Sarcasm Attribution",
            fontsize=10.5, color='#38BDF8', ha='center', va='center')

    # Draw Stage Group Containers (Subsystem Boundaries)
    def draw_group(x, y, w, h, label, stroke="#334155", bg="#131D31"):
        grp = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor=bg, edgecolor=stroke, linewidth=1.2, linestyle='--', zorder=1)
        ax.add_patch(grp)
        ax.text(x + 0.15, y + h - 0.22, label.upper(), fontsize=8.5, fontweight='bold', color=stroke, zorder=2)

    # 1. Input Stage
    draw_group(0.2, 7.2, 3.8, 3.0, "1. Audio & Text Acquisition Layer", stroke="#64748B")
    draw_rounded_box(ax, 0.5, 8.8, 3.2, 1.1, "Speech Audio Stream", "16kHz Mono PCM Audio (Microphone / WAV)", "INPUT MODALITY A (VOICE)", "#1E293B", "#38BDF8")
    draw_rounded_box(ax, 0.5, 7.4, 3.2, 1.1, "Typed Code-Mixed Text", "Hinglish / Tanglish / Teluglish String", "INPUT MODALITY B (TEXT)", "#1E293B", "#818CF8")

    # 2. Speech & Linguistic Normalization Pipeline
    draw_group(4.5, 5.8, 4.0, 4.4, "2. ASR, LID & Transliteration", stroke="#38BDF8")
    draw_rounded_box(ax, 4.8, 8.8, 3.4, 1.1, "Module 1: Neural ASR", "Whisper-small (19 Guided Languages + MPS)", "SPEECH RECOGNITION", "#1E293B", "#38BDF8")
    draw_rounded_box(ax, 4.8, 7.3, 3.4, 1.1, "Module 2: Token LID", "Unicode Block + XLM-R Token Classifier", "LANGUAGE IDENTIFICATION", "#1E293B", "#A855F7")
    draw_rounded_box(ax, 4.8, 5.95, 3.4, 1.1, "Module 3: Normalization", "indic-transliteration + Dravidian Mappings", "CANONICALIZATION", "#1E293B", "#EC4899")

    # 3. Translation & Kinship Anchor
    draw_group(9.0, 5.8, 3.8, 4.4, "3. Neural MT & Kinship Guard", stroke="#10B981")
    draw_rounded_box(ax, 9.3, 8.8, 3.2, 1.1, "Module 4: Neural MT", "Meta NLLB-200-distilled-600M Head", "NEURAL TRANSLATION", "#1E293B", "#10B981")
    draw_rounded_box(ax, 9.3, 7.3, 3.2, 1.1, "Kinship Integrity Anchor", "6-Language Familial Lexicon Matrix", "HALLUCINATION GUARD", "#1E293B", "#F59E0B")
    draw_rounded_box(ax, 9.3, 5.95, 3.2, 1.1, "Pronunciation Guide", "IPA & Roman Phonetic Engine", "ACCESSIBILITY", "#1E293B", "#14B8A6")

    # 4. Multimodal Affective Subsystem (Emotion & Sarcasm)
    draw_group(0.2, 0.8, 8.3, 4.4, "4. Multimodal Emotion, Acoustic Prosody & Sarcasm Detection", stroke="#F59E0B")
    draw_rounded_box(ax, 0.5, 3.8, 3.6, 1.1, "Module 5b: Acoustic Emotion", "Wav2Vec2-SUPERB (F0, Centroid, Harmonics)", "ACOUSTIC PROSODY", "#1E293B", "#F59E0B")
    draw_rounded_box(ax, 0.5, 2.3, 3.6, 1.1, "Module 5a: Text Emotion", "XLM-RoBERTa 7-Class Emotion Softmax", "TEXT SEMANTICS", "#1E293B", "#818CF8")
    draw_rounded_box(ax, 0.5, 0.95, 3.6, 1.1, "Module 5c: Late Fusion", "Calibrated Softmax: 0.55 Text + 0.45 Audio", "MULTIMODAL FUSION", "#1E293B", "#E11D48")

    draw_rounded_box(ax, 4.6, 2.3, 3.6, 2.6, "Module 6: Sarcasm Engine",
                     "Cross-Modal Incongruity Engine\n• Praise-Blame Paradox Check\n• Sentiment Polarity Divergence\n• D_KL + Pitch Spike (Threshold >= 0.40)",
                     "PRAGMATIC REASONING", "#1E293B", "#EC4899")

    # 5. Explainable AI (XAI)
    draw_group(9.0, 0.8, 3.8, 4.4, "5. Explainable AI (XAI) Layer", stroke="#818CF8")
    draw_rounded_box(ax, 9.3, 3.8, 3.2, 1.1, "Token Attribution (IG)", "Captum Integrated Gradients Saliency", "XAI TEXT ATTRIBUTION", "#1E293B", "#818CF8")
    draw_rounded_box(ax, 9.3, 2.3, 3.2, 1.1, "Acoustic Cue Analyzer", "Pitch Variance Windows & Formant Stability", "XAI PROSODIC EVIDENCE", "#1E293B", "#38BDF8")
    draw_rounded_box(ax, 9.3, 0.95, 3.2, 1.1, "Natural Language XAI", "Automated Justification Narrative Generator", "HUMAN-INTERPRETABLE", "#1E293B", "#10B981")

    # 6. Final Unified Output
    draw_group(13.4, 0.8, 3.8, 9.4, "6. Structured API & Presentation", stroke="#10B981")
    draw_rounded_box(ax, 13.7, 5.8, 3.2, 4.1, "Unified JSON API Contract",
                     "• transcript: Raw User Text\n• language_segments: Word Tags\n• detected_dialect: Granular Dialect\n• translated_text: Target Output\n• pronunciation_guide: Phonetics\n• emotion: Fused Label + Agreement\n• sarcasm: Incongruity + Reason\n• explanation: Top Tokens + Cues",
                     "STANDARDIZED CONTRACT", "#1E293B", "#10B981")

    draw_rounded_box(ax, 13.7, 1.1, 3.2, 4.2, "Bhasha Interactive Frontend",
                     "• Real-Time Dual Speaker Arena\n• Live WebAudio Visualizer Bars\n• Instant Code-Mixed Translation\n• Emotion & Sarcasm Glow Badges\n• XAI Diagnostic Drawer Display\n• One-Click Pronunciation Playback\n• Speaker Role Swapping",
                     "CLIENT RENDERING", "#1E293B", "#06B6D4")

    # Connective Arrows
    # Audio -> ASR
    draw_arrow(ax, (3.7, 9.35), (4.8, 9.35), label="Raw WAV", color="#38BDF8")
    # Typed -> LID / Normalization
    draw_arrow(ax, (3.7, 7.95), (4.8, 7.95), label="Text", color="#818CF8")
    # ASR -> LID
    draw_arrow(ax, (6.5, 8.8), (6.5, 8.4), label="Transcript", color="#38BDF8")
    # LID -> Normalization
    draw_arrow(ax, (6.5, 7.3), (6.5, 7.05), label="Segments", color="#A855F7")
    # Normalization -> Neural MT
    draw_arrow(ax, (8.2, 6.5), (9.3, 9.0), label="Canonical Script", color="#EC4899", rad=-0.2)
    # MT -> Kinship Guard
    draw_arrow(ax, (10.9, 8.8), (10.9, 8.4), color="#10B981")
    # Kinship -> Pronunciation
    draw_arrow(ax, (10.9, 7.3), (10.9, 7.05), color="#F59E0B")

    # Audio -> Acoustic Emotion (Module 5b)
    draw_arrow(ax, (1.8, 8.8), (1.8, 4.9), label="16kHz Audio", color="#F59E0B")
    # Text (from Normalization) -> Text Emotion (Module 5a)
    draw_arrow(ax, (4.8, 6.2), (3.8, 3.4), label="Clean Text", color="#818CF8", rad=0.2)

    # 5a & 5b -> 5c Fusion
    draw_arrow(ax, (1.8, 3.8), (1.8, 2.05), label="Acoustic Softmax", color="#F59E0B")
    draw_arrow(ax, (2.3, 2.3), (2.3, 2.05), label="Text Softmax", color="#818CF8")

    # 5a & 5b -> Sarcasm (Module 6)
    draw_arrow(ax, (3.6, 4.2), (4.6, 4.0), label="Pitch & Energy", color="#F59E0B")
    draw_arrow(ax, (3.6, 2.8), (4.6, 2.8), label="Polarity", color="#818CF8")

    # Fusion -> Sarcasm
    draw_arrow(ax, (4.1, 1.5), (4.6, 2.4), label="Incongruity", color="#E11D48", rad=-0.15)

    # Modules -> XAI
    draw_arrow(ax, (8.2, 3.5), (9.3, 4.2), label="Text Embeddings", color="#818CF8")
    draw_arrow(ax, (3.6, 4.5), (9.3, 2.9), label="Prosodic Contours", color="#38BDF8", rad=-0.25)
    draw_arrow(ax, (8.2, 2.5), (9.3, 1.5), label="Sarcasm Diagnostic", color="#EC4899", rad=-0.1)

    # MT & XAI -> Final Output JSON
    draw_arrow(ax, (12.5, 7.8), (13.7, 7.8), label="Translation", color="#10B981")
    draw_arrow(ax, (12.5, 2.5), (13.7, 6.2), label="Metrics & XAI", color="#818CF8", rad=0.1)

    # JSON -> Frontend Display
    draw_arrow(ax, (15.3, 5.8), (15.3, 5.3), label="HTTP / WebSocket", color="#06B6D4")

    # Save to file
    abs_out = os.path.abspath(output_path)
    plt.tight_layout()
    plt.savefig(abs_out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Success] Architecture diagram saved to: {abs_out}")
    return abs_out

if __name__ == "__main__":
    create_architecture_diagram()
