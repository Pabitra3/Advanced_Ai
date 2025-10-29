# app.py — Intellexa (Enhanced with Voice Interview Analysis)
import streamlit as st
import os
import requests
import json
import tempfile
from datetime import datetime
from gtts import gTTS
import matplotlib.pyplot as plt
import pandas as pd
import re
import networkx as nx
from fpdf import FPDF
from dotenv import load_dotenv
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import cv2
from fer.fer import FER
import mediapipe as mp
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import librosa
import soundfile as sf
from pydub import AudioSegment
import wave

# =============================
# Intellexa AI Tutor Settings
# =============================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = "gpt-4o-mini"
IMAGEGEN_API_KEY = os.getenv("IMAGEGEN_API_KEY")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🧠 Intellexa AI Tutor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main .block-container{padding-top:1rem;}
.stButton>button{background-color:#4CAF50;color:white;font-weight:bold;height:3em;width:100%;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Intellexa — AI Learning & Interview Coach")
st.caption("Personalized AI Tutor + Voice-Based AI Interview Coach + 3D Visual Learning + Real Video Interview Analysis")

# -------------------------
# Sidebar Inputs
# -------------------------
with st.sidebar:
    st.header("🎯 User Settings")
    name = st.text_input("Enter your name", "")
    level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
    hours = st.slider("Study Time (hours/day)", 1, 6, 2)
    days = st.slider("Number of Days for Completion", 5, 30, 10)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "📚 Learning Plan",
    "🤖 AI Tutor",
    "🎤 AI Interview Coach",
    "📊 Progress Dashboard",
    "⚡ AI Doubt Visualizer"
])

# =========================
# Helper functions
# =========================
def call_openrouter(prompt, timeout=40):
    """Call OpenRouter chat completions and return text or raise Exception."""
    try:
        if not OPENROUTER_API_KEY:
            raise Exception("OPENROUTER_API_KEY not found. Please set it in .env file")
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL_ID, 
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=timeout
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error {r.status_code}: {r.text}")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '_audio.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        st.warning(f"Audio extraction warning: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio to text using speech recognition"""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        st.warning(f"Transcription warning: {e}")
        return None

def analyze_speech_patterns(audio_path):
    """Analyze speech patterns: pace, pauses, filler words"""
    try:
        y, sr_rate = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr_rate)
        
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.3
        speech_frames = rms > threshold
        
        frame_length = len(y) / len(rms)
        speaking_time = np.sum(speech_frames) * frame_length / sr_rate
        pause_time = duration - speaking_time
        
        speech_rate = speaking_time / duration if duration > 0 else 0
        
        if speech_rate > 0.8:
            pace = "Fast"
        elif speech_rate > 0.6:
            pace = "Moderate"
        else:
            pace = "Slow"
        
        return {
            "duration": duration,
            "speaking_time": speaking_time,
            "pause_time": pause_time,
            "speech_rate": speech_rate,
            "pace": pace,
            "pauses_count": np.sum(np.diff(speech_frames.astype(int)) == -1)
        }
    except Exception as e:
        st.warning(f"Speech analysis warning: {e}")
        return None

def analyze_voice_interview(audio_path, question, domain):
    """
    Comprehensive voice/audio analysis
    Returns detailed metrics including transcription and speech patterns
    """
    try:
        analysis_results = {
            "transcript": "",
            "speech_patterns": {},
            "word_count": 0,
            "filler_words_count": 0,
            "clarity_score": 0
        }
        
        # 1. TRANSCRIBE AUDIO
        st.info("🎯 Transcribing audio...")
        transcript = transcribe_audio(audio_path)
        
        if transcript:
            analysis_results["transcript"] = transcript
            analysis_results["word_count"] = len(transcript.split())
            
            # Count filler words
            filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally']
            transcript_lower = transcript.lower()
            filler_count = sum(transcript_lower.count(filler) for filler in filler_words)
            analysis_results["filler_words_count"] = filler_count
        else:
            st.warning("⚠️ Could not transcribe audio. Analysis will be limited.")
        
        # 2. ANALYZE SPEECH PATTERNS
        st.info("🎯 Analyzing speech patterns...")
        speech_analysis = analyze_speech_patterns(audio_path)
        
        if speech_analysis:
            analysis_results["speech_patterns"] = speech_analysis
        
        # 3. CALCULATE CLARITY SCORE (0-10)
        # Based on: speech rate, pauses, filler words
        clarity_score = 10
        
        if speech_analysis:
            # Penalize for too fast or too slow pace
            if speech_analysis['pace'] == 'Fast':
                clarity_score -= 2
            elif speech_analysis['pace'] == 'Slow':
                clarity_score -= 1
            
            # Penalize for excessive pauses
            if speech_analysis['pauses_count'] > 10:
                clarity_score -= 1
        
        # Penalize for filler words
        if analysis_results["word_count"] > 0:
            filler_ratio = analysis_results["filler_words_count"] / analysis_results["word_count"]
            if filler_ratio > 0.1:
                clarity_score -= 2
            elif filler_ratio > 0.05:
                clarity_score -= 1
        
        analysis_results["clarity_score"] = max(0, clarity_score)
        
        return analysis_results
        
    except Exception as e:
        raise Exception(f"Voice analysis failed: {str(e)}")

def generate_voice_feedback(question, analysis_results, domain):
    """Generate AI feedback based on voice analysis results"""
    
    transcript = analysis_results.get("transcript", "")
    speech_data = analysis_results.get("speech_patterns", {})
    word_count = analysis_results.get("word_count", 0)
    filler_count = analysis_results.get("filler_words_count", 0)
    clarity_score = analysis_results.get("clarity_score", 0)
    
    prompt = f"""
You are an expert interview coach analyzing a voice interview response.

**Interview Question:** {question}
**Domain:** {domain}
**Candidate's Answer:** {transcript if transcript else "[Audio transcription not available]"}

**VOICE ANALYSIS METRICS:**

1. **Transcription Quality:**
   - Word Count: {word_count}
   - Clarity Score: {clarity_score}/10

2. **Speech Patterns:**
   - Response Duration: {speech_data.get('duration', 0):.1f} seconds
   - Speaking Time: {speech_data.get('speaking_time', 0):.1f}s
   - Pause Time: {speech_data.get('pause_time', 0):.1f}s
   - Speech Pace: {speech_data.get('pace', 'N/A')}
   - Number of Pauses: {speech_data.get('pauses_count', 0)}

3. **Verbal Fluency:**
   - Filler Words Detected: {filler_count}
   - Fluency Assessment: {"Needs improvement" if filler_count > 5 else "Good"}

Based on this comprehensive voice analysis, provide:

**1. CONTENT ANALYSIS (Score: X/10)**
- Technical accuracy and completeness of the answer
- Relevance to the question
- Depth of knowledge demonstrated
- Structure and organization

**2. VERBAL COMMUNICATION (Score: X/10)**
- Speech clarity and articulation
- Pace and rhythm (not too fast or slow)
- Confidence in vocal delivery
- Use of technical terminology
- Filler words assessment

**3. FLUENCY & COHERENCE (Score: X/10)**
- Smooth flow of speech
- Logical progression of ideas
- Minimal hesitations and pauses
- Natural speaking style

**4. CONFIDENCE & PROFESSIONALISM (Score: X/10)**
- Vocal confidence level
- Professional tone
- Enthusiasm and engagement
- Composure under pressure

**5. OVERALL IMPRESSION (Score: X/10)**
- Interview readiness based on voice
- Would you hire based on this audio response?
- Professional presence through voice

**6. KEY STRENGTHS** (List 3-4 specific strengths observed)

**7. AREAS FOR IMPROVEMENT** (List 3-4 with actionable tips)

**8. SPECIFIC RECOMMENDATIONS**
- Speech practice exercises
- How to reduce filler words
- Pace improvement techniques
- Confidence building tips

Be specific, constructive, and reference the actual metrics provided.
"""
    
    try:
        feedback = call_openrouter(prompt, timeout=60)
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"

def analyze_video_interview(video_path, question, user_answer):
    """
    Comprehensive video analysis using CV and ML models
    Returns detailed metrics and scores
    """
    try:
        analysis_results = {
            "emotion_analysis": {},
            "eye_contact": {},
            "gestures": {},
            "facial_expressions": {},
            "speech_patterns": {},
            "transcript": "",
            "response_time": 0
        }
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        emotion_detector = FER(mtcnn=True)
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        emotions_detected = []
        eye_contact_frames = 0
        looking_away_frames = 0
        gesture_movements = []
        facial_expression_changes = 0
        frame_count = 0
        sample_rate = 5
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = min(frame_count / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing frame {frame_count}/{total_frames}...")
            
            if frame_count % sample_rate != 0:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                emotions = emotion_detector.detect_emotions(rgb_frame)
                if emotions:
                    dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                    emotions_detected.append(dominant_emotion[0])
            except:
                pass
            
            try:
                results_face = face_mesh.process(rgb_frame)
                if results_face.multi_face_landmarks:
                    face_landmarks = results_face.multi_face_landmarks[0]
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]
                    eye_y_diff = abs(left_eye.y - right_eye.y)
                    if eye_y_diff < 0.05:
                        eye_contact_frames += 1
                    else:
                        looking_away_frames += 1
            except:
                pass
            
            try:
                results_pose = pose.process(rgb_frame)
                if results_pose.pose_landmarks:
                    landmarks = results_pose.pose_landmarks.landmark
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    gesture_score = abs(left_hand.y - left_shoulder.y) + abs(right_hand.y - right_shoulder.y)
                    gesture_movements.append(gesture_score)
            except:
                pass
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        if emotions_detected:
            emotion_counts = pd.Series(emotions_detected).value_counts()
            dominant_emotion = emotion_counts.index[0]
            emotion_confidence = emotion_counts.iloc[0] / len(emotions_detected) * 100
            
            analysis_results["emotion_analysis"] = {
                "dominant_emotion": dominant_emotion,
                "confidence": emotion_confidence,
                "distribution": emotion_counts.to_dict(),
                "changes": len(set(emotions_detected))
            }
        
        total_eye_frames = eye_contact_frames + looking_away_frames
        if total_eye_frames > 0:
            eye_contact_percentage = (eye_contact_frames / total_eye_frames) * 100
            analysis_results["eye_contact"] = {
                "percentage": eye_contact_percentage,
                "maintained_frames": eye_contact_frames,
                "looking_away_frames": looking_away_frames
            }
        
        if gesture_movements:
            avg_gesture = np.mean(gesture_movements)
            gesture_variance = np.std(gesture_movements)
            
            if avg_gesture > 0.5:
                gesture_level = "Excessive"
            elif avg_gesture > 0.2:
                gesture_level = "Moderate"
            else:
                gesture_level = "Minimal"
            
            analysis_results["gestures"] = {
                "level": gesture_level,
                "average_movement": avg_gesture,
                "variance": gesture_variance
            }
        
        analysis_results["response_time"] = video_duration
        
        audio_path = extract_audio_from_video(video_path)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            if transcript:
                analysis_results["transcript"] = transcript
            
            speech_analysis = analyze_speech_patterns(audio_path)
            if speech_analysis:
                analysis_results["speech_patterns"] = speech_analysis
            
            try:
                os.remove(audio_path)
            except:
                pass
        
        if not analysis_results["transcript"] and user_answer:
            analysis_results["transcript"] = user_answer
        
        return analysis_results
        
    except Exception as e:
        raise Exception(f"Video analysis failed: {str(e)}")

def generate_comprehensive_feedback(question, analysis_results, domain):
    """Generate AI feedback based on video analysis results"""
    
    emotion_data = analysis_results.get("emotion_analysis", {})
    eye_contact_data = analysis_results.get("eye_contact", {})
    gesture_data = analysis_results.get("gestures", {})
    speech_data = analysis_results.get("speech_patterns", {})
    transcript = analysis_results.get("transcript", "")
    response_time = analysis_results.get("response_time", 0)
    
    prompt = f"""
You are an expert interview coach analyzing a video interview response.

**Interview Question:** {question}
**Domain:** {domain}
**Candidate's Answer:** {transcript if transcript else "[Video analysis without transcript]"}

**VIDEO ANALYSIS METRICS:**

1. **Emotion Analysis:**
   - Dominant Emotion: {emotion_data.get('dominant_emotion', 'N/A')}
   - Confidence: {emotion_data.get('confidence', 0):.1f}%
   - Emotional Changes: {emotion_data.get('changes', 0)}
   - Distribution: {emotion_data.get('distribution', {})}

2. **Eye Contact:**
   - Eye Contact Maintained: {eye_contact_data.get('percentage', 0):.1f}%
   - Looking Away: {100 - eye_contact_data.get('percentage', 0):.1f}%

3. **Gestures & Body Language:**
   - Gesture Level: {gesture_data.get('level', 'N/A')}
   - Movement Score: {gesture_data.get('average_movement', 0):.2f}

4. **Speech Patterns:**
   - Response Duration: {response_time:.1f} seconds
   - Speaking Time: {speech_data.get('speaking_time', 0):.1f}s
   - Pause Time: {speech_data.get('pause_time', 0):.1f}s
   - Speech Pace: {speech_data.get('pace', 'N/A')}
   - Number of Pauses: {speech_data.get('pauses_count', 0)}

Based on this comprehensive analysis, provide:

**1. CONTENT ANALYSIS (Score: X/10)**
- Evaluate the answer's technical accuracy and completeness
- Relevance to the question
- Depth of knowledge demonstrated

**2. VERBAL COMMUNICATION (Score: X/10)**
- Speech clarity and pace
- Use of technical terminology
- Confidence in delivery
- Filler words and pauses assessment

**3. NON-VERBAL COMMUNICATION (Score: X/10)**
- Eye contact quality and impact
- Facial expressions appropriateness
- Body language and posture
- Gesture effectiveness

**4. EMOTIONAL INTELLIGENCE (Score: X/10)**
- Confidence level displayed
- Stress management
- Enthusiasm and engagement
- Composure

**5. OVERALL IMPRESSION (Score: X/10)**
- Professional presence
- Interview readiness
- Hiring recommendation

**6. KEY STRENGTHS** (List 3-4 specific strengths)

**7. AREAS FOR IMPROVEMENT** (List 3-4 with actionable tips)

**8. SPECIFIC RECOMMENDATIONS**
- What to practice
- How to improve weak areas
- Exercises for better performance

Be specific, constructive, and reference the actual metrics provided.
"""
    
    try:
        feedback = call_openrouter(prompt, timeout=60)
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"

def create_pdf(plan_text, student_name, goal, filename="Learning_Plan.pdf"):
    """Generate PDF of learning plan"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Learning Plan for {student_name}", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Goal: {goal}", ln=True)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(6)
        for line in plan_text.split("\n"):
            pdf.multi_cell(0, 7, line)
        pdf.output(filename)
        return filename
    except Exception as e:
        raise Exception(f"PDF generation failed: {str(e)}")

def generate_3d_animation(concept, animation_type="rotation"):
    """
    Generate 3D animation based on the concept
    Returns the path to the saved animation GIF
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['neural', 'network', 'ai', 'ml', 'deep learning']):
            frames = []
            for angle in range(0, 360, 5):
                ax.clear()
                layers = [5, 8, 8, 3]
                z_positions = [0, 3, 6, 9]
                
                for layer_idx, (num_nodes, z) in enumerate(zip(layers, z_positions)):
                    theta = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
                    x = np.cos(theta)
                    y = np.sin(theta)
                    z_arr = np.full(num_nodes, z)
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][layer_idx]
                    ax.scatter(x, y, z_arr, c=colors, s=300, alpha=0.8, edgecolors='black', linewidth=2)
                    
                    if layer_idx < len(layers) - 1:
                        next_theta = np.linspace(0, 2*np.pi, layers[layer_idx + 1], endpoint=False)
                        next_x = np.cos(next_theta)
                        next_y = np.sin(next_theta)
                        next_z = z_positions[layer_idx + 1]
                        
                        for i in range(num_nodes):
                            for j in range(layers[layer_idx + 1]):
                                ax.plot([x[i], next_x[j]], [y[i], next_y[j]], 
                                       [z_arr[i], next_z], 'gray', alpha=0.2, linewidth=0.5)
                
                ax.set_xlabel('X', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y', fontsize=12, fontweight='bold')
                ax.set_zlabel('Layer Depth', fontsize=12, fontweight='bold')
                ax.set_title(f'Neural Network Architecture\n{concept}', fontsize=14, fontweight='bold', pad=20)
                ax.view_init(elev=20, azim=angle)
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-1, 10])
                
                frame_path = f'temp_frame_{len(frames)}.png'
                plt.savefig(frame_path, dpi=80, bbox_inches='tight')
                frames.append(frame_path)
            
        else:
            frames = []
            for angle in range(0, 360, 5):
                ax.clear()
                t = np.linspace(0, 4*np.pi, 100)
                x = np.cos(t) * (1 + t/10)
                y = np.sin(t) * (1 + t/10)
                z = t
                
                for i in range(len(t)-1):
                    color = plt.cm.plasma(i / len(t))
                    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                           color=color, linewidth=2.5, alpha=0.8)
                
                key_points = [0, 25, 50, 75, 99]
                ax.scatter(x[key_points], y[key_points], z[key_points], 
                          c='red', s=150, alpha=0.9, edgecolors='black', linewidth=2)
                
                ax.set_xlabel('X Dimension', fontsize=12, fontweight='bold')
                ax.set_ylabel('Y Dimension', fontsize=12, fontweight='bold')
                ax.set_zlabel('Progress/Time', fontsize=12, fontweight='bold')
                ax.set_title(f'Concept Visualization\n{concept}', fontsize=14, fontweight='bold', pad=20)
                ax.view_init(elev=20, azim=angle)
                
                frame_path = f'temp_frame_{len(frames)}.png'
                plt.savefig(frame_path, dpi=80, bbox_inches='tight')
                frames.append(frame_path)
        
        plt.close()
        
        # Create GIF from frames
        import imageio
        images = [imageio.imread(frame) for frame in frames]
        gif_path = 'concept_animation.gif'
        imageio.mimsave(gif_path, images, duration=0.1)
        
        # Clean up frames
        for frame in frames:
            if os.path.exists(frame):
                os.remove(frame)
        
        return gif_path
        
    except Exception as e:
        raise Exception(f"3D Animation generation failed: {str(e)}")

# =========================
# Tab 1: Learning Plan
# =========================
with tabs[0]:
    st.subheader("📅 AI-Powered Personalized Learning Plan")
    goal = st.selectbox(
        "Choose your Goal",
        ["Data Analytics","Web Development","Machine Learning","Data Science",
         "MERN Stack","Java Development","Android Development"]
    )

    if "ai_plan_text" not in st.session_state:
        st.session_state["ai_plan_text"] = None

    if st.button("✨ Generate AI Learning Plan"):
        if not name:
            st.warning("Please enter your name in the sidebar first.")
        else:
            with st.spinner("AI is generating your personalized study plan..."):
                prompt = f"""
                You are an expert AI tutor.
                Create a personalized {days}-day learning plan for a student named {name}
                who wants to master {goal}.
                The student's skill level is {level}, and they can study {hours} hours per day.

                The plan must be returned as a numbered list in this format:
                Day 01: [Topic Name] — [Key Concepts/Activities]
                Day 02: [Topic Name] — [Key Concepts/Activities]
                ...
                Up to Day {days}.
                Be detailed and tailored to their skill level and available time.
                """
                try:
                    ai_plan = call_openrouter(prompt, timeout=60)
                    st.session_state["ai_plan_text"] = ai_plan
                    st.success("✅ AI learning plan generated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to generate plan: {e}")

    if st.session_state.get("ai_plan_text"):
        plan_text = st.session_state["ai_plan_text"]
        st.markdown("#### Your AI-generated plan:")
        st.code(plan_text, language="")

        day_blocks = re.findall(r"(Day\s*\d{1,2}[:\-]?\s*)([^\n]+)", plan_text)
        if day_blocks:
            st.markdown("### 🗓️ Click a Day to View Details")
            if "plan_progress" not in st.session_state:
                st.session_state["plan_progress"] = {}
            for i, (day, content) in enumerate(day_blocks):
                label = day.strip().replace(":", "")
                with st.expander(f"{label}"):
                    st.markdown(f"**{label}** — {content.strip()}")
                    done = st.checkbox(f"✅ Mark {label} as Done", key=f"done_{i}")
                    st.session_state["plan_progress"][label] = bool(done)

            if st.button("📄 Download Learning Plan as PDF"):
                try:
                    pdf_path = create_pdf(plan_text, name or "Student", goal)
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_path))
                except Exception as e:
                    st.error(f"Failed to create PDF: {e}")

# =========================
# Tab 2: AI Tutor (Adaptive)
# =========================
with tabs[1]:
    st.subheader("🤖 AI Tutor Assistant")
    if "ai_plan_text" in st.session_state and st.session_state["ai_plan_text"]:
        topic_list = re.findall(r"Day\s*\d{1,2}[:\-]?\s*(.+?)(?:—|-|\n|$)", st.session_state["ai_plan_text"])
        topic_list = [t.strip() for t in topic_list if t.strip()]
    else:
        topic_list = ["General Concepts", "Exercises", "Mini Project"]

    chosen_topic = st.selectbox("Select a Topic", topic_list)
    action = st.radio(
        "Choose AI Action",
        ["Explain Topic Clearly", "Generate Practice Questions", "Suggest Next Topic", "Give Study Improvement Tips"]
    )

    if st.button("Ask AI ✨", key="ai_tutor"):
        if not name:
            st.warning("Please enter your name in the sidebar first.")
        else:
            with st.spinner("AI is generating..."):
                style = {
                    "Beginner": "Use simple analogies, step-by-step examples, small hands-on exercises.",
                    "Intermediate": "Provide conceptual depth, practical tips, and intermediate exercises.",
                    "Advanced": "Give in-depth technical explanation, advanced examples, and references."
                }[level]
                prompt = f"You are an AI tutor for {goal}. {action} about {chosen_topic} for a {level} learner. {style}"
                try:
                    result = call_openrouter(prompt, timeout=40)
                    st.markdown("### AI Response")
                    st.write(result)
                except Exception as e:
                    st.error(f"AI request failed: {e}")

# =========================
# Tab 3: AI Interview Coach (ENHANCED with Voice + Video)
# =========================
with tabs[2]:
    st.title("🎯 AI Interview Coach with Voice & Video Analysis")
    st.write("Practice interviews with comprehensive AI feedback - Text, Voice, or Video!")

    interview_mode = st.radio(
        "Choose Interview Mode:",
        ["📝 Text-Based Interview", "🎙️ Voice Interview (Audio Analysis)", "🎥 Video Interview (Full Analysis)"],
        horizontal=True
    )

    num_questions = st.number_input(
        "Enter number of questions (Maximum: 150)", 
        min_value=1, 
        max_value=150, 
        value=3, 
        step=1,
        help="AI can generate up to 150 interview questions at once"
    )
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("Choose your interview domain", 
                             ["General", "Cybersecurity", "AI/ML", "Web Development", "Data Science"])
    
    with col2:
        difficulty_level = st.selectbox(
            "Select difficulty level",
            ["Beginner", "Intermediate", "Advanced", "Expert"],
            index=1,
            help="Choose the difficulty level for interview questions"
        )

    if st.button("🧩 Generate Interview Questions"):
        with st.spinner("Generating interview questions..."):
            try:
                q_prompt = f"Generate {num_questions} realistic {domain} interview questions. Format them as Q1:, Q2:, etc."
                interview_questions = call_openrouter(q_prompt)
                st.session_state["questions"] = [q.strip() for q in interview_questions.split("\n") 
                                                if q.strip().startswith("Q")]
                st.success(f"✅ Generated {len(st.session_state['questions'])} questions!")
            except Exception as e:
                st.error(f"Failed to generate questions: {e}")

    if "questions" in st.session_state:
        # TEXT-BASED INTERVIEW MODE
        if interview_mode == "📝 Text-Based Interview":
            for idx, question in enumerate(st.session_state["questions"], 1):
                st.markdown(f"### 🗣️ {question}")
                
                user_answer = st.text_area("Type your answer:", key=f"text_{idx}")
                if st.button(f"Evaluate Answer {idx}", key=f"eval_{idx}"):
                    with st.spinner("Analyzing your response..."):
                        try:
                            eval_prompt = f"""
                            You are an interview coach. Evaluate the answer below for the question: "{question}"
                            Candidate's Answer: "{user_answer}"
                            Provide:
                            - Technical Accuracy (0-10)
                            - Clarity (0-10)
                            - Confidence (0-10)
                            - Professionalism (0-10)
                            - Short improvement advice
                            """
                            feedback = call_openrouter(eval_prompt)
                            st.markdown("### 🧩 AI Feedback")
                            st.write(feedback)
                            
                            if "interview_answers" not in st.session_state:
                                st.session_state["interview_answers"] = {}
                            st.session_state["interview_answers"][idx] = {
                                "question": question,
                                "answer": user_answer,
                                "feedback": feedback
                            }
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
        
        # VOICE INTERVIEW MODE (NEW)
        elif interview_mode == "🎙️ Voice Interview (Audio Analysis)":
            st.markdown("---")
            st.markdown("### 🎙️ Voice Interview Analysis")
            st.info("""
            **🎤 Upload your audio response and get AI analysis:**
            - 🗣️ Speech Transcription
            - ⏱️ Speech Pace & Rhythm Analysis
            - 🎯 Clarity & Fluency Assessment
            - 📊 Filler Words Detection
            - 💬 Pause Pattern Analysis
            - 🎓 Content Quality Evaluation
            """)
            
            for idx, question in enumerate(st.session_state["questions"], 1):
                with st.expander(f"🎤 Question {idx}: Voice Interview", expanded=(idx==1)):
                    st.markdown(f"#### 🗣️ {question}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**📤 Upload Your Audio Response**")
                        audio_file = st.file_uploader(
                            f"Upload audio recording (Q{idx})",
                            type=["wav", "mp3", "m4a", "ogg", "flac"],
                            key=f"audio_upload_{idx}",
                            help="Record yourself answering the question and upload the audio file"
                        )
                        
                        if audio_file is not None:
                            st.audio(audio_file)
                            
                            # Save audio temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                                tmp_audio.write(audio_file.read())
                                audio_path = tmp_audio.name
                            
                            st.success("✅ Audio uploaded successfully!")
                    
                    with col2:
                        st.markdown("**⚙️ Analysis Info**")
                        st.info("""
                        AI will analyze:
                        - Speech clarity
                        - Speaking pace
                        - Filler words
                        - Pauses
                        - Content quality
                        """)
                    
                    st.markdown("---")
                    st.markdown("**💡 Tips for best results:**")
                    st.caption("• Speak clearly and at a moderate pace")
                    st.caption("• Minimize background noise")
                    st.caption("• Answer comprehensively")
                    
                    if st.button(f"🔍 Analyze Voice Interview {idx}", key=f"analyze_voice_{idx}"):
                        if audio_file is None:
                            st.warning("⚠️ Please upload an audio file first.")
                        else:
                            st.markdown("---")
                            st.markdown("### 🤖 AI Voice Analysis in Progress...")
                            
                            try:
                                # Perform voice analysis
                                with st.spinner("🎤 Analyzing your voice response... This may take 30-60 seconds..."):
                                    analysis_results = analyze_voice_interview(
                                        audio_path,
                                        question,
                                        domain
                                    )
                                
                                st.success("✅ Voice analysis complete!")
                                
                                # Display Analysis Results
                                st.markdown("---")
                                st.markdown("## 📊 Voice Analysis Results")
                                
                                # Create metrics display
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    word_count = analysis_results.get("word_count", 0)
                                    st.metric("Word Count", word_count)
                                
                                with col2:
                                    speech_data = analysis_results.get("speech_patterns", {})
                                    duration = speech_data.get("duration", 0)
                                    st.metric("Duration", f"{duration:.1f}s")
                                
                                with col3:
                                    pace = speech_data.get("pace", "N/A")
                                    pace_icon = "🐇" if pace == "Fast" else "🐢" if pace == "Slow" else "✅"
                                    st.metric("Speech Pace", f"{pace_icon} {pace}")
                                
                                with col4:
                                    clarity_score = analysis_results.get("clarity_score", 0)
                                    st.metric("Clarity Score", f"{clarity_score}/10")
                                
                                # Detailed breakdowns
                                st.markdown("---")
                                tab1, tab2, tab3 = st.tabs(["📝 Transcript", "🗣️ Speech Analysis", "📊 Metrics"])
                                
                                with tab1:
                                    st.subheader("Transcription")
                                    transcript = analysis_results.get("transcript", "")
                                    if transcript:
                                        st.success("✅ Transcription successful!")
                                        st.info(transcript)
                                        
                                        # Word cloud style analysis
                                        st.markdown("**📈 Quick Stats:**")
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write(f"• Total words: {word_count}")
                                            st.write(f"• Sentences: ~{len(transcript.split('.'))-1}")
                                        with col_b:
                                            filler_count = analysis_results.get("filler_words_count", 0)
                                            st.write(f"• Filler words: {filler_count}")
                                            if word_count > 0:
                                                filler_pct = (filler_count / word_count) * 100
                                                st.write(f"• Filler ratio: {filler_pct:.1f}%")
                                    else:
                                        st.warning("⚠️ Transcription not available. Analysis will be limited.")
                                
                                with tab2:
                                    st.subheader("Speech Pattern Analysis")
                                    if speech_data:
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            st.metric("Speaking Time", f"{speech_data.get('speaking_time', 0):.1f}s")
                                            st.metric("Speech Pace", speech_data.get('pace', 'N/A'))
                                            st.metric("Pauses Count", speech_data.get('pauses_count', 0))
                                        
                                        with col_b:
                                            st.metric("Pause Time", f"{speech_data.get('pause_time', 0):.1f}s")
                                            st.metric("Speech Rate", f"{speech_data.get('speech_rate', 0):.0%}")
                                        
                                        # Visualize speech vs pause time
                                        st.markdown("**Time Distribution:**")
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        labels = ['Speaking', 'Pauses']
                                        sizes = [speech_data.get('speaking_time', 0), speech_data.get('pause_time', 0)]
                                        colors = ['#4CAF50', '#FFC107']
                                        explode = (0.1, 0)
                                        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                              autopct='%1.1f%%', shadow=True, startangle=90)
                                        ax.set_title('Speech Time Distribution', fontweight='bold')
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                        # Pace feedback
                                        pace = speech_data.get('pace', '')
                                        if pace == "Fast":
                                            st.warning("⚠️ You're speaking too fast. Slow down for better clarity.")
                                        elif pace == "Moderate":
                                            st.success("✅ Perfect speaking pace!")
                                        else:
                                            st.info("💡 Try to speak with more energy and confidence.")
                                    else:
                                        st.info("No speech pattern data available")
                                
                                with tab3:
                                    st.subheader("Performance Metrics")
                                    
                                    # Filler words analysis
                                    filler_count = analysis_results.get("filler_words_count", 0)
                                    st.markdown("**🎯 Filler Words Analysis:**")
                                    if filler_count == 0:
                                        st.success("✅ Excellent! No filler words detected.")
                                    elif filler_count <= 3:
                                        st.info(f"👍 Good! Only {filler_count} filler words detected.")
                                    elif filler_count <= 7:
                                        st.warning(f"⚠️ Moderate: {filler_count} filler words detected. Try to reduce them.")
                                    else:
                                        st.error(f"❌ Too many filler words ({filler_count}). Practice speaking more fluently.")
                                    
                                    # Clarity score visualization
                                    st.markdown("**📊 Overall Clarity Score:**")
                                    clarity = clarity_score
                                    
                                    fig, ax = plt.subplots(figsize=(10, 2))
                                    ax.barh(['Clarity'], [clarity], color='#4CAF50' if clarity >= 7 else '#FFC107' if clarity >= 5 else '#FF5722')
                                    ax.set_xlim(0, 10)
                                    ax.set_xlabel('Score', fontweight='bold')
                                    ax.set_title(f'Voice Clarity Score: {clarity}/10', fontweight='bold')
                                    for i, v in enumerate([clarity]):
                                        ax.text(v + 0.2, i, str(v), va='center', fontweight='bold')
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Generate AI Feedback
                                st.markdown("---")
                                st.markdown("## 🤖 AI Expert Feedback")
                                
                                with st.spinner("Generating comprehensive AI feedback based on voice analysis..."):
                                    feedback = generate_voice_feedback(
                                        question,
                                        analysis_results,
                                        domain
                                    )
                                    st.markdown(feedback)
                                
                                # Extract scores for visualization
                                scores = {}
                                score_categories = [
                                    "CONTENT ANALYSIS", "VERBAL COMMUNICATION",
                                    "FLUENCY & COHERENCE", "CONFIDENCE & PROFESSIONALISM",
                                    "OVERALL IMPRESSION"
                                ]
                                
                                for category in score_categories:
                                    match = re.search(f"{category}.*?Score:.*?(\\d+)/10", feedback, re.IGNORECASE)
                                    if match:
                                        short_name = category.replace("ANALYSIS", "").replace("COMMUNICATION", "COMM").replace("&", "").strip()
                                        scores[short_name] = int(match.group(1))
                                
                                if scores:
                                    st.markdown("---")
                                    st.markdown("### 📈 Performance Radar Chart")
                                    
                                    categories = list(scores.keys())
                                    values = list(scores.values())
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                                    
                                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                                    values_plot = values + values[:1]
                                    angles_plot = angles + angles[:1]
                                    
                                    ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#FF6B6B')
                                    ax.fill(angles_plot, values_plot, alpha=0.25, color='#FF6B6B')
                                    ax.set_xticks(angles)
                                    ax.set_xticklabels(categories, size=9)
                                    ax.set_ylim(0, 10)
                                    ax.set_yticks([2, 4, 6, 8, 10])
                                    ax.set_yticklabels(['2', '4', '6', '8', '10'])
                                    ax.grid(True)
                                    ax.set_title(f"Voice Interview Performance - Q{idx}",
                                               size=14, fontweight='bold', pad=20)
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    avg_score = np.mean(values)
                                    st.metric("🎯 Overall Performance Score", f"{avg_score:.1f}/10")
                                    
                                    if avg_score >= 8:
                                        st.success("🌟 Excellent Voice Performance! You're interview-ready!")
                                    elif avg_score >= 6:
                                        st.info("👍 Good Performance! A few improvements will make you outstanding.")
                                    else:
                                        st.warning("💪 Keep Practicing! Focus on the improvement areas.")
                                
                                # Store results
                                if "voice_interview_results" not in st.session_state:
                                    st.session_state["voice_interview_results"] = {}
                                
                                st.session_state["voice_interview_results"][idx] = {
                                    "question": question,
                                    "transcript": transcript,
                                    "feedback": feedback,
                                    "scores": scores,
                                    "analysis_results": analysis_results,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # Download report
                                st.markdown("---")
                                if st.button(f"📄 Download Voice Analysis Report Q{idx}", key=f"download_voice_report_{idx}"):
                                    try:
                                        report_text = f"""
VOICE INTERVIEW ANALYSIS REPORT
================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Candidate: {name if name else 'User'}
Domain: {domain}

QUESTION {idx}: {question}

TRANSCRIPT:
{transcript if transcript else 'Not available'}

VOICE ANALYSIS METRICS:
-----------------------
Word Count: {word_count}
Duration: {duration:.1f} seconds
Speech Pace: {pace}
Clarity Score: {clarity_score}/10
Filler Words: {analysis_results.get("filler_words_count", 0)}

Speech Patterns:
  - Speaking Time: {speech_data.get('speaking_time', 0):.1f}s
  - Pause Time: {speech_data.get('pause_time', 0):.1f}s
  - Pauses Count: {speech_data.get('pauses_count', 0)}

AI EXPERT FEEDBACK:
-------------------
{feedback}

PERFORMANCE SCORES:
{json.dumps(scores, indent=2)}

RECOMMENDATIONS:
----------------
Review the detailed feedback above and practice the suggested improvements.
Focus on reducing filler words and maintaining a steady pace.
                                        """
                                        
                                        st.download_button(
                                            label="⬇️ Download Voice Analysis Report",
                                            data=report_text,
                                            file_name=f"Voice_Interview_Q{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                            mime="text/plain",
                                            key=f"download_voice_btn_{idx}"
                                        )
                                    except Exception as e:
                                        st.error(f"Report generation error: {e}")
                                
                                # Clean up audio file
                                try:
                                    os.remove(audio_path)
                                except:
                                    pass
                                    
                            except Exception as e:
                                st.error(f"❌ Voice analysis failed: {str(e)}")
                                st.info("💡 Make sure the audio is clear. Supported formats: WAV, MP3, M4A, OGG, FLAC")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Voice Interview Summary
            if "voice_interview_results" in st.session_state and st.session_state["voice_interview_results"]:
                st.markdown("---")
                st.markdown("## 📊 Voice Interview Session Summary")
                
                results = st.session_state["voice_interview_results"]
                
                all_scores = []
                for result in results.values():
                    if result.get("scores"):
                        all_scores.append(result["scores"])
                
                if all_scores:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📝 Questions Analyzed", len(results))
                    with col2:
                        overall_avg = np.mean([np.mean(list(s.values())) for s in all_scores])
                        st.metric("🎯 Average Score", f"{overall_avg:.1f}/10")
                    with col3:
                        readiness = "Ready ✅" if overall_avg >= 7 else "Keep Practicing 💪"
                        st.metric("🚀 Interview Readiness", readiness)
        
        # VIDEO INTERVIEW MODE
        else:
            st.markdown("---")
            st.markdown("### 🎥 Real Video Interview Analysis")
            st.info("""
            **📹 Upload your video and get comprehensive AI analysis:**
            - 🎭 Real-time Emotion Detection
            - 👁️ Eye Contact Tracking  
            - 🙋 Gesture & Posture Analysis
            - 🗣️ Speech Pattern Recognition
            - 😊 Facial Expression Analysis
            - ⏱️ Response Time Tracking
            """)
            
            for idx, question in enumerate(st.session_state["questions"], 1):
                with st.expander(f"🎤 Question {idx}: Video Interview", expanded=(idx==1)):
                    st.markdown(f"#### 🗣️ {question}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        video_file = st.file_uploader(
                            f"📤 Upload your video response (Q{idx})",
                            type=["mp4", "avi", "mov", "webm"],
                            key=f"video_upload_{idx}",
                            help="Record yourself answering the question and upload the video"
                        )
                        
                        if video_file is not None:
                            st.video(video_file)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                                tmp_video.write(video_file.read())
                                video_path = tmp_video.name
                            
                            st.success("✅ Video uploaded successfully!")
                    
                    with col2:
                        st.markdown("**⚙️ Analysis Settings**")
                        st.info("All features enabled by default")
                        analyze_all = st.checkbox("Full Analysis", value=True, key=f"full_{idx}")
                    
                    st.markdown("---")
                    video_transcript = st.text_area(
                        "📝 Optional: Provide transcript to improve analysis accuracy",
                        placeholder="Type what you said in the video (optional)...",
                        key=f"transcript_{idx}",
                        height=100
                    )
                    
                    if st.button(f"🔍 Analyze Video Interview {idx}", key=f"analyze_video_{idx}"):
                        if video_file is None:
                            st.warning("⚠️ Please upload a video first.")
                        else:
                            st.markdown("---")
                            st.markdown("### 🤖 AI Video Analysis in Progress...")
                            
                            try:
                                with st.spinner("🎬 Analyzing video... This may take 1-2 minutes..."):
                                    analysis_results = analyze_video_interview(
                                        video_path,
                                        question,
                                        video_transcript
                                    )
                                
                                st.success("✅ Video analysis complete!")
                                
                                st.markdown("---")
                                st.markdown("## 📊 Video Analysis Results")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    emotion = analysis_results.get("emotion_analysis", {}).get("dominant_emotion", "N/A")
                                    emotion_icons = {
                                        "happy": "😊",
                                        "neutral": "😐",
                                        "sad": "😢",
                                        "angry": "😠",
                                        "surprise": "😲",
                                        "fear": "😨"
                                    }
                                    icon = emotion_icons.get(emotion.lower(), "😊")
                                    st.metric("Dominant Emotion", f"{icon} {emotion.title()}")
                                
                                with col2:
                                    eye_contact = analysis_results.get("eye_contact", {}).get("percentage", 0)
                                    st.metric("Eye Contact", f"{eye_contact:.1f}%")
                                
                                with col3:
                                    gesture_level = analysis_results.get("gestures", {}).get("level", "N/A")
                                    st.metric("Gesture Activity", gesture_level)
                                
                                with col4:
                                    response_time = analysis_results.get("response_time", 0)
                                    st.metric("Response Time", f"{response_time:.1f}s")
                                
                                st.markdown("---")
                                tab1, tab2, tab3, tab4 = st.tabs(["😊 Emotions", "👁️ Eye Contact", "🙋 Gestures", "🗣️ Speech"])
                                
                                with tab1:
                                    st.subheader("Emotion Analysis")
                                    emotion_data = analysis_results.get("emotion_analysis", {})
                                    if emotion_data:
                                        st.write(f"**Dominant Emotion:** {emotion_data.get('dominant_emotion', 'N/A').title()}")
                                        st.write(f"**Confidence:** {emotion_data.get('confidence', 0):.1f}%")
                                        st.write(f"**Emotional Changes:** {emotion_data.get('changes', 0)}")
                                        
                                        distribution = emotion_data.get('distribution', {})
                                        if distribution:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            emotions = list(distribution.keys())
                                            counts = list(distribution.values())
                                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
                                            ax.bar(emotions, counts, color=colors[:len(emotions)])
                                            ax.set_xlabel('Emotion', fontweight='bold')
                                            ax.set_ylabel('Frequency', fontweight='bold')
                                            ax.set_title('Emotion Distribution During Interview', fontweight='bold')
                                            st.pyplot(fig)
                                            plt.close()
                                    else:
                                        st.info("No emotion data available")
                                
                                with tab2:
                                    st.subheader("Eye Contact Analysis")
                                    eye_data = analysis_results.get("eye_contact", {})
                                    if eye_data:
                                        eye_pct = eye_data.get('percentage', 0)
                                        st.write(f"**Eye Contact Maintained:** {eye_pct:.1f}%")
                                        st.write(f"**Looking Away:** {100-eye_pct:.1f}%")
                                        
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        labels = ['Eye Contact', 'Looking Away']
                                        sizes = [eye_pct, 100-eye_pct]
                                        colors = ['#4CAF50', '#FFC107']
                                        explode = (0.1, 0)
                                        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                              autopct='%1.1f%%', shadow=True, startangle=90)
                                        ax.set_title('Eye Contact Distribution', fontweight='bold')
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                        if eye_pct >= 70:
                                            st.success("✅ Excellent eye contact!")
                                        elif eye_pct >= 50:
                                            st.info("👍 Good eye contact, but can be improved")
                                        else:
                                            st.warning("⚠️ Need to maintain better eye contact")
                                    else:
                                        st.info("No eye contact data available")
                                
                                with tab3:
                                    st.subheader("Gesture & Body Language")
                                    gesture_data = analysis_results.get("gestures", {})
                                    if gesture_data:
                                        st.write(f"**Gesture Level:** {gesture_data.get('level', 'N/A')}")
                                        st.write(f"**Movement Score:** {gesture_data.get('average_movement', 0):.2f}")
                                        
                                        level = gesture_data.get('level', '')
                                        if level == "Excessive":
                                            st.warning("⚠️ Too much movement - try to stay more composed")
                                        elif level == "Moderate":
                                            st.success("✅ Appropriate level of gestures")
                                        else:
                                            st.info("💡 You can use more hand gestures to emphasize points")
                                    else:
                                        st.info("No gesture data available")
                                
                                with tab4:
                                    st.subheader("Speech Pattern Analysis")
                                    speech_data = analysis_results.get("speech_patterns", {})
                                    transcript = analysis_results.get("transcript", "")
                                    
                                    if transcript:
                                        st.write("**📝 Transcript:**")
                                        st.info(transcript)
                                    
                                    if speech_data:
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Speaking Time", f"{speech_data.get('speaking_time', 0):.1f}s")
                                            st.metric("Speech Pace", speech_data.get('pace', 'N/A'))
                                        with col_b:
                                            st.metric("Pause Time", f"{speech_data.get('pause_time', 0):.1f}s")
                                            st.metric("Pauses Count", speech_data.get('pauses_count', 0))
                                        
                                        pace = speech_data.get('pace', '')
                                        if pace == "Fast":
                                            st.warning("⚠️ Speaking too fast - slow down for clarity")
                                        elif pace == "Moderate":
                                            st.success("✅ Good speaking pace")
                                        else:
                                            st.info("💡 Try to speak with more confidence and energy")
                                    else:
                                        st.info("No speech pattern data available")
                                
                                # Generate AI Feedback
                                st.markdown("---")
                                st.markdown("## 🤖 AI Expert Feedback")
                                
                                with st.spinner("Generating comprehensive AI feedback..."):
                                    feedback = generate_comprehensive_feedback(
                                        question,
                                        analysis_results,
                                        domain
                                    )
                                    st.markdown(feedback)
                                
                                # Extract scores for visualization
                                scores = {}
                                score_categories = [
                                    "CONTENT ANALYSIS", "VERBAL COMMUNICATION",
                                    "NON-VERBAL COMMUNICATION", "EMOTIONAL INTELLIGENCE",
                                    "OVERALL IMPRESSION"
                                ]
                                
                                for category in score_categories:
                                    match = re.search(f"{category}.*?Score:.*?(\\d+)/10", feedback, re.IGNORECASE)
                                    if match:
                                        short_name = category.replace("ANALYSIS", "").replace("COMMUNICATION", "COMM").strip()
                                        scores[short_name] = int(match.group(1))
                                
                                if scores:
                                    st.markdown("---")
                                    st.markdown("### 📈 Performance Radar Chart")
                                    
                                    categories = list(scores.keys())
                                    values = list(scores.values())
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                                    
                                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                                    values_plot = values + values[:1]
                                    angles_plot = angles + angles[:1]
                                    
                                    ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#4CAF50')
                                    ax.fill(angles_plot, values_plot, alpha=0.25, color='#4CAF50')
                                    ax.set_xticks(angles)
                                    ax.set_xticklabels(categories, size=10)
                                    ax.set_ylim(0, 10)
                                    ax.set_yticks([2, 4, 6, 8, 10])
                                    ax.set_yticklabels(['2', '4', '6', '8', '10'])
                                    ax.grid(True)
                                    ax.set_title(f"Interview Performance Analysis - Q{idx}",
                                               size=14, fontweight='bold', pad=20)
                                    
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    avg_score = np.mean(values)
                                    st.metric("🎯 Overall Performance Score", f"{avg_score:.1f}/10")
                                    
                                    if avg_score >= 8:
                                        st.success("🌟 Excellent Performance! You're interview-ready!")
                                    elif avg_score >= 6:
                                        st.info("👍 Good Performance! A few improvements will make you outstanding.")
                                    else:
                                        st.warning("💪 Keep Practicing! Focus on the improvement areas.")
                                
                                # Store results
                                if "video_interview_results" not in st.session_state:
                                    st.session_state["video_interview_results"] = {}
                                
                                st.session_state["video_interview_results"][idx] = {
                                    "question": question,
                                    "transcript": transcript,
                                    "feedback": feedback,
                                    "scores": scores,
                                    "analysis_results": analysis_results,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # Download report
                                st.markdown("---")
                                if st.button(f"📄 Download Detailed Report Q{idx}", key=f"download_report_{idx}"):
                                    try:
                                        report_text = f"""
VIDEO INTERVIEW ANALYSIS REPORT
================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Candidate: {name if name else 'User'}
Domain: {domain}

QUESTION {idx}: {question}

TRANSCRIPT:
{transcript if transcript else 'Not available'}

VIDEO ANALYSIS METRICS:
-----------------------
Emotion Analysis:
  - Dominant Emotion: {analysis_results.get('emotion_analysis', {}).get('dominant_emotion', 'N/A')}
  - Confidence: {analysis_results.get('emotion_analysis', {}).get('confidence', 0):.1f}%
  
Eye Contact:
  - Maintained: {analysis_results.get('eye_contact', {}).get('percentage', 0):.1f}%
  
Gestures:
  - Level: {analysis_results.get('gestures', {}).get('level', 'N/A')}
  
Speech Patterns:
  - Response Time: {analysis_results.get('response_time', 0):.1f}s
  - Pace: {analysis_results.get('speech_patterns', {}).get('pace', 'N/A')}

AI EXPERT FEEDBACK:
-------------------
{feedback}

PERFORMANCE SCORES:
{json.dumps(scores, indent=2)}

RECOMMENDATIONS:
----------------
Review the detailed feedback above and practice the suggested improvements.
                                        """
                                        
                                        st.download_button(
                                            label="⬇️ Download Analysis Report",
                                            data=report_text,
                                            file_name=f"Interview_Analysis_Q{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                            mime="text/plain",
                                            key=f"download_btn_{idx}"
                                        )
                                    except Exception as e:
                                        st.error(f"Report generation error: {e}")
                                
                                # Clean up video file
                                try:
                                    os.remove(video_path)
                                except:
                                    pass
                                    
                            except Exception as e:
                                st.error(f"❌ Video analysis failed: {str(e)}")
                                st.info("💡 Make sure the video is clear and contains your face. Try uploading a different video.")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Video Interview Summary
            if "video_interview_results" in st.session_state and st.session_state["video_interview_results"]:
                st.markdown("---")
                st.markdown("## 📊 Video Interview Session Summary")
                
                results = st.session_state["video_interview_results"]
                
                all_scores = []
                for result in results.values():
                    if result.get("scores"):
                        all_scores.append(result["scores"])
                
                if all_scores:
                    avg_scores = {}
                    for score_dict in all_scores:
                        for key, value in score_dict.items():
                            if key not in avg_scores:
                                avg_scores[key] = []
                            avg_scores[key].append(value)
                    
                    final_avg = {k: np.mean(v) for k, v in avg_scores.items()}
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📝 Questions Analyzed", len(results))
                    with col2:
                        overall_avg = np.mean([np.mean(list(s.values())) for s in all_scores])
                        st.metric("🎯 Average Score", f"{overall_avg:.1f}/10")
                    with col3:
                        readiness = "Ready ✅" if overall_avg >= 7 else "Keep Practicing 💪"
                        st.metric("🚀 Interview Readiness", readiness)
                    
                    # Progress chart across questions
                    st.markdown("### 📈 Performance Across Questions")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for metric in final_avg.keys():
                        metric_values = []
                        for q_num in sorted(results.keys()):
                            if results[q_num].get("scores") and metric in results[q_num]["scores"]:
                                metric_values.append(results[q_num]["scores"][metric])
                            else:
                                metric_values.append(None)
                        
                        if any(v is not None for v in metric_values):
                            ax.plot(sorted(results.keys()), metric_values, marker='o',
                                   label=metric, linewidth=2.5, markersize=8)
                    
                    ax.set_xlabel('Question Number', fontweight='bold', fontsize=12)
                    ax.set_ylabel('Score (0-10)', fontweight='bold', fontsize=12)
                    ax.set_title('Performance Improvement Over Time', fontweight='bold', fontsize=14, pad=20)
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 10)
                    st.pyplot(fig)
                    plt.close()

# =========================
# Tab 4: Progress Dashboard
# =========================
with tabs[3]:
    st.subheader("📊 Progress Dashboard")
    
    completed = st.session_state.get("plan_progress", {})
    if completed:
        completed_days = [d for d, v in completed.items() if v]
        total_days = len(completed)
        if completed_days:
            st.markdown(f"**✅ Completed Days:** {', '.join(completed_days)}")
            st.progress(len(completed_days) / total_days if total_days > 0 else 0)
        else:
            st.info("No days marked completed yet.")
    else:
        st.info("No learning plan yet. Generate one in the Learning Plan tab.")

    history = st.session_state.get("interview_answers", {})
    if history:
        st.markdown("### 📝 Text Interview History")
        df = pd.DataFrame(history).T
        st.dataframe(df[["question", "answer", "feedback"]])

        def extract_score(text, metric):
            match = re.search(f"{metric}: *(\\d+)", text, re.IGNORECASE)
            return int(match.group(1)) if match else None

        df["Confidence"] = df["feedback"].apply(lambda x: extract_score(x, "confidence"))
        df["Technical"] = df["feedback"].apply(lambda x: extract_score(x, "technical"))
        df["Clarity"] = df["feedback"].apply(lambda x: extract_score(x, "clarity"))

        st.markdown("### 📈 Text Interview Score Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        if df["Confidence"].notnull().any():
            ax.plot(df.index, df["Confidence"], marker="o", label="Confidence", linewidth=2)
        if df["Technical"].notnull().any():
            ax.plot(df.index, df["Technical"], marker="o", label="Technical", linewidth=2)
        if df["Clarity"].notnull().any():
            ax.plot(df.index, df["Clarity"], marker="o", label="Clarity", linewidth=2)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Question Number")
        ax.set_ylabel("Score (0-10)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Voice Interview Analytics
    voice_results = st.session_state.get("voice_interview_results", {})
    if voice_results:
        st.markdown("---")
        st.markdown("### 🎙️ Voice Interview Analytics")
        
        voice_data = []
        for q_num, result in voice_results.items():
            if result.get("scores"):
                row = {"Question": q_num, "Timestamp": result.get("timestamp", "N/A")}
                row.update(result["scores"])
                voice_data.append(row)
        
        if voice_data:
            voice_df = pd.DataFrame(voice_data)
            st.dataframe(voice_df)
            
            st.markdown("### 📊 Voice Interview Performance Comparison")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = [col for col in voice_df.columns if col not in ["Question", "Timestamp"]]
            x = np.arange(len(voice_df))
            width = 0.15
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for i, metric in enumerate(metrics):
                if metric in voice_df.columns:
                    offset = width * (i - len(metrics)/2)
                    ax.bar(x + offset, voice_df[metric], width,
                          label=metric, color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_xlabel('Question Number', fontweight='bold')
            ax.set_ylabel('Score (0-10)', fontweight='bold')
            ax.set_title('Voice Interview Performance Across Questions', fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{q}" for q in voice_df["Question"]])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 10)
            
            st.pyplot(fig)
            plt.close()
    
    # Video Interview Analytics
    video_results = st.session_state.get("video_interview_results", {})
    if video_results:
        st.markdown("---")
        st.markdown("### 🎥 Video Interview Analytics")
        
        video_data = []
        for q_num, result in video_results.items():
            if result.get("scores"):
                row = {"Question": q_num, "Timestamp": result.get("timestamp", "N/A")}
                row.update(result["scores"])
                video_data.append(row)
        
        if video_data:
            video_df = pd.DataFrame(video_data)
            st.dataframe(video_df)
            
            st.markdown("### 📊 Video Interview Performance Comparison")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = [col for col in video_df.columns if col not in ["Question", "Timestamp"]]
            x = np.arange(len(video_df))
            width = 0.15
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            for i, metric in enumerate(metrics):
                if metric in video_df.columns:
                    offset = width * (i - len(metrics)/2)
                    ax.bar(x + offset, video_df[metric], width,
                          label=metric, color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_xlabel('Question Number', fontweight='bold')
            ax.set_ylabel('Score (0-10)', fontweight='bold')
            ax.set_title('Video Interview Performance Across Questions', fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{q}" for q in video_df["Question"]])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 10)
            
            st.pyplot(fig)
            plt.close()
    
    # Combined Achievements
    if history or voice_results or video_results:
        st.markdown("---")
        st.markdown("### 🏅 Achievements & Badges")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if history and len(history) >= 1:
                st.success("🥇 First Text Interview!")
            if voice_results and len(voice_results) >= 1:
                st.success("🎙️ First Voice Interview!")
            if video_results and len(video_results) >= 1:
                st.success("🎥 First Video Interview!")
        
        with col2:
            if history:
                df = pd.DataFrame(history).T
                if "Confidence" in df.columns and df["Confidence"].dropna().max() >= 8:
                    st.success("💡 Confidence Master!")
                if "Technical" in df.columns and df["Technical"].dropna().max() >= 8:
                    st.success("🧠 Technical Expert!")
            
            if voice_results:
                all_voice_scores = []
                for result in voice_results.values():
                    if result.get("scores"):
                        all_voice_scores.extend(result["scores"].values())
                
                if all_voice_scores:
                    avg_voice = np.mean(all_voice_scores)
                    if avg_voice >= 8:
                        st.success("🎤 Voice Interview Pro!")
        
        with col3:
            if video_results:
                all_video_scores = []
                for result in video_results.values():
                    if result.get("scores"):
                        all_video_scores.extend(result["scores"].values())
                
                if all_video_scores:
                    avg_video = np.mean(all_video_scores)
                    if avg_video >= 8:
                        st.success("⭐ Video Interview Pro!")
                    if len(video_results) >= 3:
                        st.success("🎬 Persistent Practitioner!")
            
            # Multi-format achievement
            formats_used = sum([bool(history), bool(voice_results), bool(video_results)])
            if formats_used >= 2:
                st.success("🌟 Multi-Format Master!")
            if formats_used == 3:
                st.success("🏆 Complete Interview Champion!")
    
    if not history and not voice_results and not video_results:
        st.info("No interview data yet. Use the AI Interview Coach tab to get started!")

# =========================
# Tab 5: AI Doubt Visualizer (ENHANCED WITH 3D)
# =========================
with tabs[4]:
    st.header("🎨 AI Doubt Visualizer")
    st.write("Get visual explanations")

    doubt = st.text_area("💭 Enter your doubt or concept to visualize", 
                         placeholder="e.g., Neural Networks, Binary Search Tree, Data Pipeline, etc.")
    
    if st.button("🎨 Generate Visual Explanation"):
        if not doubt.strip():
            st.warning("Please enter a concept to visualize.")
        else:
            with st.spinner("🎬 Generating comprehensive visual explanation"):
                try:
                    # Step 1: Get AI explanation
                    visual_prompt = f"""
                    Create a detailed step-by-step visual explanation for this concept: {doubt}
                    
                    Format your response as:
                    1. Main Concept: [brief description]
                    2. Key Components: [list 3-5 main parts]
                    3. How It Works: [step-by-step process]
                    4. Visual Representation: [describe a simple diagram structure]
                    5. Simple Analogy: [relate to everyday concept]
                    6. Practical Application: [real-world use case]
                    
                    Make it clear, beginner-friendly, and easy to visualize.
                    """
                    
                    explanation = call_openrouter(visual_prompt, timeout=60)
                    
                    # Display explanation
                    st.markdown("### 🧠 AI Visual Explanation")
                    st.info(explanation)
                    
                    # Step 2: Extract and visualize "Visual Representation" section with ENHANCED ACCURACY
                    try:
                        # Parse the explanation to extract Visual Representation section
                        visual_rep_content = ""
                        lines = explanation.split('\n')
                        in_visual_section = False
                        
                        for line in lines:
                            line_stripped = line.strip()
                            if 'visual representation' in line_stripped.lower() and ':' in line_stripped:
                                in_visual_section = True
                                # Get content after the colon
                                visual_rep_content = line_stripped.split(':', 1)[1].strip()
                                continue
                            elif in_visual_section:
                                # Check if we hit the next section
                                if any(header in line_stripped.lower() for header in ['simple analogy:', 'practical application:', '5.', '6.']):
                                    break
                                if line_stripped:
                                    visual_rep_content += " " + line_stripped
                        
                        if visual_rep_content:
                            # Use AI to generate a more accurate diagram specification
                            diagram_spec_prompt = f"""
                            Based on this visual representation description for "{doubt}":
                            
                            "{visual_rep_content}"
                            
                            Generate a detailed diagram specification with:
                            1. DIAGRAM_TYPE: (flowchart/network/hierarchy/cycle/components/layers)
                            2. NUM_ELEMENTS: (number between 3-8)
                            3. ELEMENT_LABELS: (comma-separated list of specific labels for each element)
                            4. CONNECTIONS: (describe how elements connect, e.g., "sequential", "hub-spoke", "layered")
                            5. MAIN_CONCEPT: (central concept in 2-3 words)
                            
                            Format strictly as:
                            DIAGRAM_TYPE: [type]
                            NUM_ELEMENTS: [number]
                            ELEMENT_LABELS: [label1, label2, label3, ...]
                            CONNECTIONS: [description]
                            MAIN_CONCEPT: [concept]
                            
                            Be specific and match the concept accurately.
                            """
                            
                            diagram_spec = call_openrouter(diagram_spec_prompt, timeout=40)
                            
                            # Parse the specification
                            spec_dict = {}
                            for line in diagram_spec.split('\n'):
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    spec_dict[key.strip().upper()] = value.strip()
                            
                            # Extract parsed data with fallbacks
                            diagram_type = spec_dict.get('DIAGRAM_TYPE', 'components').lower()
                            
                            try:
                                num_elements = int(spec_dict.get('NUM_ELEMENTS', '5'))
                                num_elements = max(3, min(8, num_elements))  # Clamp between 3-8
                            except:
                                num_elements = 5
                            
                            # Parse element labels
                            labels_str = spec_dict.get('ELEMENT_LABELS', '')
                            if labels_str:
                                element_labels = [label.strip() for label in labels_str.split(',')]
                                # Ensure we have enough labels
                                while len(element_labels) < num_elements:
                                    element_labels.append(f"Element {len(element_labels) + 1}")
                                element_labels = element_labels[:num_elements]
                            else:
                                element_labels = [f"Element {i+1}" for i in range(num_elements)]
                            
                            connections = spec_dict.get('CONNECTIONS', 'sequential').lower()
                            main_concept = spec_dict.get('MAIN_CONCEPT', doubt)[:30]
                            
                            # Remove any type prefixes from diagram_type
                            for prefix in ['flowchart', 'network', 'hierarchy', 'cycle', 'components', 'layers']:
                                if prefix in diagram_type:
                                    diagram_type = prefix
                                    break
                            st.markdown("---")
                            st.markdown("### 🎨 Enhanced Visual Representation Diagram")
                            st.info(f"📊 Accurate diagram for: **{main_concept}** (Type: {diagram_type.title()})")
                            
                            # Create enhanced visualization
                            fig = plt.figure(figsize=(16, 12))
                            fig.patch.set_facecolor('#F5F7FA')
                            
                            # Main title with better formatting
                            fig.suptitle(f'Visual Representation: {main_concept}', 
                                        fontsize=22, fontweight='bold', y=0.97,
                                        color='#1a1a1a', bbox=dict(boxstyle='round,pad=0.5', 
                                                                    facecolor='#E8F4F8', 
                                                                    edgecolor='#2196F3', linewidth=2))
                            
                            # Create main visualization area
                            ax = fig.add_axes([0.08, 0.12, 0.84, 0.78])
                            ax.axis('off')
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            
                            # Generate visualization based on accurate diagram type
                            if diagram_type == 'flowchart' or 'flow' in diagram_type:
                                # ENHANCED FLOWCHART
                                ax.text(0.5, 0.95, '🔄 Process Flow Diagram', 
                                       ha='center', fontsize=16, fontweight='bold',
                                       color='#1976D2', bbox=dict(boxstyle='round,pad=0.5',
                                                                  facecolor='white',
                                                                  edgecolor='#1976D2', linewidth=2))
                                
                                y_positions = np.linspace(0.80, 0.15, num_elements)
                                colors = plt.cm.Blues(np.linspace(0.5, 0.9, num_elements))
                                
                                for i, (y_pos, color, label) in enumerate(zip(y_positions, colors, element_labels)):
                                    # Determine box style based on position
                                    if i == 0:
                                        box_style = 'round,pad=0.02'  # Start
                                    elif i == num_elements - 1:
                                        box_style = 'round,pad=0.02'  # End
                                    else:
                                        box_style = 'square,pad=0.02'  # Process
                                    
                                    # Draw enhanced box with shadow effect
                                    shadow = plt.Rectangle((0.18, y_pos - 0.035), 0.65, 0.07,
                                                          facecolor='gray', alpha=0.2,
                                                          transform=ax.transData, zorder=1)
                                    ax.add_patch(shadow)
                                    
                                    box = plt.Rectangle((0.17, y_pos - 0.03), 0.65, 0.07,
                                                       facecolor=color, edgecolor='#0D47A1',
                                                       linewidth=2.5, transform=ax.transData,
                                                       zorder=2)
                                    ax.add_patch(box)
                                    
                                    # Add step number badge
                                    badge = plt.Circle((0.15, y_pos), 0.025,
                                                      facecolor='#FF6B35', edgecolor='white',
                                                      linewidth=2, zorder=3)
                                    ax.add_patch(badge)
                                    ax.text(0.15, y_pos, str(i+1), ha='center', va='center',
                                           fontsize=10, fontweight='bold', color='white', zorder=4)
                                    
                                    # Add label with word wrapping
                                    wrapped_label = '\n'.join([label[j:j+35] for j in range(0, len(label), 35)])
                                    ax.text(0.495, y_pos, wrapped_label, ha='center', va='center',
                                           fontsize=10, fontweight='bold', color='white',
                                           zorder=3)
                                    
                                    # Draw arrow to next step
                                    if i < num_elements - 1:
                                        ax.annotate('', xy=(0.495, y_positions[i+1] + 0.04),
                                                   xytext=(0.495, y_pos - 0.04),
                                                   arrowprops=dict(arrowstyle='->', lw=4,
                                                                 color='#333333',
                                                                 connectionstyle="arc3,rad=0"),
                                                   zorder=1)
                            
                            elif diagram_type == 'network' or 'node' in diagram_type or 'graph' in diagram_type:
                                # ENHANCED NETWORK/GRAPH
                                ax.text(0.5, 0.95, '🕸️ Network Architecture', 
                                       ha='center', fontsize=16, fontweight='bold',
                                       color='#E65100', bbox=dict(boxstyle='round,pad=0.5',
                                                                  facecolor='white',
                                                                  edgecolor='#E65100', linewidth=2))
                                
                                center_x, center_y = 0.5, 0.5
                                
                                # Enhanced central node with gradient effect
                                for r in [0.13, 0.11, 0.09]:
                                    alpha_val = 0.3 if r == 0.13 else 0.5 if r == 0.11 else 1.0
                                    central_circle = plt.Circle((center_x, center_y), r,
                                                              facecolor='#FF6B35', 
                                                              edgecolor='#BF360C' if r == 0.09 else 'none',
                                                              linewidth=3, alpha=alpha_val, zorder=3)
                                    ax.add_patch(central_circle)
                                
                                ax.text(center_x, center_y, main_concept[:15],
                                       ha='center', va='center', fontsize=11,
                                       fontweight='bold', color='white', zorder=4,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6B35',
                                                edgecolor='none'))
                                
                                # Surrounding nodes with better spacing
                                angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False)
                                radius = 0.32
                                colors = plt.cm.Oranges(np.linspace(0.5, 0.85, num_elements))
                                
                                for i, (angle, color, label) in enumerate(zip(angles, colors, element_labels)):
                                    x = center_x + radius * np.cos(angle)
                                    y = center_y + radius * np.sin(angle)
                                    
                                    # Draw connection with varying thickness
                                    ax.plot([center_x, x], [center_y, y],
                                           color='#666666', linewidth=3, alpha=0.4,
                                           zorder=1, linestyle='--')
                                    
                                    # Enhanced node with shadow
                                    shadow_circle = plt.Circle((x + 0.01, y - 0.01), 0.08,
                                                             facecolor='gray', alpha=0.3, zorder=1)
                                    ax.add_patch(shadow_circle)
                                    
                                    node_circle = plt.Circle((x, y), 0.08,
                                                           facecolor=color, edgecolor='#E65100',
                                                           linewidth=2.5, zorder=2)
                                    ax.add_patch(node_circle)
                                    
                                    # Wrap label text
                                    wrapped_label = '\n'.join([label[j:j+12] for j in range(0, len(label), 12)])
                                    ax.text(x, y, wrapped_label, ha='center', va='center',
                                           fontsize=8, fontweight='bold', color='white', zorder=3)
                                    
                                    # Add label below node
                                    label_y = y - 0.12
                                    ax.text(x, label_y, f"Node {i+1}", ha='center', va='top',
                                           fontsize=8, color='#333333',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                                    edgecolor='#E65100', linewidth=1))
                            
                            elif diagram_type == 'hierarchy' or 'layer' in diagram_type:
                                # ENHANCED HIERARCHICAL/LAYERED
                                ax.text(0.5, 0.95, '📊 Hierarchical Structure', 
                                       ha='center', fontsize=16, fontweight='bold',
                                       color='#6A1B9A', bbox=dict(boxstyle='round,pad=0.5',
                                                                  facecolor='white',
                                                                  edgecolor='#6A1B9A', linewidth=2))
                                
                                y_positions = np.linspace(0.78, 0.15, num_elements)
                                widths = np.linspace(0.75, 0.25, num_elements)
                                colors = plt.cm.RdPu(np.linspace(0.4, 0.85, num_elements))
                                
                                for i, (y_pos, width, color, label) in enumerate(zip(y_positions, widths, colors, element_labels)):
                                    x_start = (1 - width) / 2
                                    
                                    # Shadow effect
                                    shadow = plt.Rectangle((x_start + 0.01, y_pos - 0.01), width, 0.09,
                                                          facecolor='gray', alpha=0.2, zorder=1)
                                    ax.add_patch(shadow)
                                    
                                    # Main layer with gradient-like effect
                                    rect = plt.Rectangle((x_start, y_pos), width, 0.09,
                                                        facecolor=color, edgecolor='#4A148C',
                                                        linewidth=2.5, zorder=2)
                                    ax.add_patch(rect)
                                    
                                    # Add tier indicator
                                    tier_text = f"Tier {i+1}" if i < num_elements - 1 else "Base"
                                    ax.text(x_start - 0.08, y_pos + 0.045, tier_text,
                                           ha='right', va='center', fontsize=9,
                                           fontweight='bold', color='#4A148C',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                                    edgecolor='#6A1B9A', linewidth=1.5))
                                    
                                    # Wrap and add label
                                    wrapped_label = '\n'.join([label[j:j+40] for j in range(0, len(label), 40)])
                                    ax.text(x_start + width/2, y_pos + 0.045, wrapped_label,
                                           ha='center', va='center', fontsize=10,
                                           fontweight='bold', color='white', zorder=3)
                                    
                                    # Draw connecting lines to next layer
                                    if i < num_elements - 1:
                                        next_x_start = (1 - widths[i+1]) / 2
                                        next_width = widths[i+1]
                                        next_y = y_positions[i+1]
                                        
                                        # Left connection
                                        ax.plot([x_start, next_x_start], [y_pos, next_y + 0.09],
                                               'k--', linewidth=1.5, alpha=0.4, zorder=1)
                                        # Right connection
                                        ax.plot([x_start + width, next_x_start + next_width],
                                               [y_pos, next_y + 0.09],
                                               'k--', linewidth=1.5, alpha=0.4, zorder=1)
                            
                            elif diagram_type == 'cycle' or 'circular' in diagram_type or 'loop' in diagram_type:
                                # ENHANCED CIRCULAR/CYCLE
                                ax.text(0.5, 0.95, '🔁 Cyclic Process', 
                                       ha='center', fontsize=16, fontweight='bold',
                                       color='#00796B', bbox=dict(boxstyle='round,pad=0.5',
                                                                  facecolor='white',
                                                                  edgecolor='#00796B', linewidth=2))
                                
                                center_x, center_y = 0.5, 0.48
                                radius = 0.32
                                angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False) + np.pi/2
                                colors = plt.cm.Greens(np.linspace(0.5, 0.9, num_elements))
                                
                                # Draw circular path with gradient
                                circle_angles = np.linspace(0, 2*np.pi, 100)
                                for i in range(len(circle_angles)-1):
                                    color_idx = int(i / len(circle_angles) * len(colors))
                                    x1 = center_x + radius * np.cos(circle_angles[i])
                                    y1 = center_y + radius * np.sin(circle_angles[i])
                                    x2 = center_x + radius * np.cos(circle_angles[i+1])
                                    y2 = center_y + radius * np.sin(circle_angles[i+1])
                                    ax.plot([x1, x2], [y1, y2], color=colors[color_idx % len(colors)],
                                           linewidth=4, alpha=0.3, zorder=1)
                                
                                for i, (angle, color, label) in enumerate(zip(angles, colors, element_labels)):
                                    x = center_x + radius * np.cos(angle)
                                    y = center_y + radius * np.sin(angle)
                                    
                                    # Shadow effect
                                    shadow_circle = plt.Circle((x + 0.01, y - 0.01), 0.09,
                                                             facecolor='gray', alpha=0.3, zorder=1)
                                    ax.add_patch(shadow_circle)
                                    
                                    # Main node
                                    node_circle = plt.Circle((x, y), 0.09,
                                                           facecolor=color, edgecolor='#004D40',
                                                           linewidth=3, zorder=2)
                                    ax.add_patch(node_circle)
                                    
                                    # Step number
                                    ax.text(x, y + 0.03, str(i+1), ha='center', va='center',
                                           fontsize=12, fontweight='bold', color='white', zorder=3)
                                    
                                    # Wrap label
                                    wrapped_label = '\n'.join([label[j:j+10] for j in range(0, len(label), 10)])
                                    ax.text(x, y - 0.03, wrapped_label[:20], ha='center', va='center',
                                           fontsize=7, fontweight='bold', color='white', zorder=3)
                                    
                                    # Directional arrow to next node
                                    next_idx = (i + 1) % num_elements
                                    next_angle = angles[next_idx]
                                    next_x = center_x + radius * np.cos(next_angle)
                                    next_y = center_y + radius * np.sin(next_angle)
                                    
                                    # Calculate arrow position on circle edge
                                    mid_angle = (angle + next_angle) / 2
                                    arrow_x = center_x + radius * np.cos(mid_angle)
                                    arrow_y = center_y + radius * np.sin(mid_angle)
                                    
                                    # Direction vector
                                    dx = next_x - x
                                    dy = next_y - y
                                    
                                    ax.annotate('', xy=(arrow_x + dx*0.1, arrow_y + dy*0.1),
                                               xytext=(arrow_x - dx*0.1, arrow_y - dy*0.1),
                                               arrowprops=dict(arrowstyle='->', lw=4,
                                                             color='#004D40'),
                                               zorder=2)
                            
                            else:
                                # ENHANCED COMPONENT BLOCKS
                                ax.text(0.5, 0.95, '🔷 Component Architecture', 
                                       ha='center', fontsize=16, fontweight='bold',
                                       color='#1565C0', bbox=dict(boxstyle='round,pad=0.5',
                                                                  facecolor='white',
                                                                  edgecolor='#1565C0', linewidth=2))
                                
                                cols = min(3, num_elements)
                                rows = (num_elements + cols - 1) // cols
                                
                                colors = plt.cm.Set3(np.linspace(0, 1, num_elements))
                                
                                component_idx = 0
                                for row in range(rows):
                                    actual_cols = min(cols, num_elements - row * cols)
                                    start_x = 0.5 - (actual_cols * 0.28) / 2
                                    
                                    for col in range(actual_cols):
                                        if component_idx >= num_elements:
                                            break
                                        
                                        x = start_x + col * 0.30
                                        y = 0.72 - row * 0.28
                                        
                                        label = element_labels[component_idx]
                                        color = colors[component_idx]
                                        
                                        # Shadow
                                        shadow = plt.Rectangle((x + 0.01, y - 0.01), 0.24, 0.18,
                                                              facecolor='gray', alpha=0.2, zorder=1)
                                        ax.add_patch(shadow)
                                        
                                        # Component box with rounded corners
                                        rect = plt.Rectangle((x, y), 0.24, 0.18,
                                                            facecolor=color, edgecolor='#0D47A1',
                                                            linewidth=2.5, zorder=2)
                                        ax.add_patch(rect)
                                        
                                        # Component number badge
                                        badge = plt.Circle((x + 0.02, y + 0.16), 0.018,
                                                          facecolor='#0D47A1', edgecolor='white',
                                                          linewidth=2, zorder=3)
                                        ax.add_patch(badge)
                                        ax.text(x + 0.02, y + 0.16, str(component_idx + 1),
                                               ha='center', va='center', fontsize=8,
                                               fontweight='bold', color='white', zorder=4)
                                        
                                        # Wrap label
                                        wrapped_label = '\n'.join([label[j:j+15] for j in range(0, len(label), 15)])
                                        ax.text(x + 0.12, y + 0.09, wrapped_label[:45],
                                               ha='center', va='center', fontsize=9,
                                               fontweight='bold', color='#1a1a1a', zorder=3)
                                        
                                        # Draw connections between adjacent components
                                        if col < actual_cols - 1 and component_idx < num_elements - 1:
                                            ax.plot([x + 0.24, x + 0.30], [y + 0.09, y + 0.09],
                                                   'k-', linewidth=2, alpha=0.3, zorder=1)
                                        
                                        component_idx += 1
                            
                            # Enhanced description box
                            desc_box = plt.Rectangle((0.03, 0.01), 0.94, 0.09,
                                                    facecolor='white', alpha=0.95,
                                                    edgecolor='#2196F3', linewidth=2.5,
                                                    zorder=10)
                            ax.add_patch(desc_box)
                            
                            # Add icon
                            ax.text(0.04, 0.055, '📝', ha='left', va='center',
                                   fontsize=16, zorder=11)
                            
                            # Add description with better formatting
                            max_desc_length = 280
                            display_desc = visual_rep_content[:max_desc_length]
                            if len(visual_rep_content) > max_desc_length:
                                display_desc = display_desc.rsplit(' ', 1)[0] + "..."
                            
                            # Wrap text
                            words = display_desc.split()
                            lines = []
                            current_line = ""
                            for word in words:
                                if len(current_line + " " + word) <= 90:
                                    current_line += (" " + word if current_line else word)
                                else:
                                    lines.append(current_line)
                                    current_line = word
                            if current_line:
                                lines.append(current_line)
                            
                            y_text = 0.075
                            for line in lines[:3]:
                                ax.text(0.5, y_text, line, ha='center', va='center',
                                       fontsize=8.5, color='#333333', zorder=11)
                                y_text -= 0.02
                            
                            # Enhanced footer
                            footer_box = plt.Rectangle((0.03, -0.005), 0.94, 0.01,
                                                      facecolor='#E3F2FD', alpha=0.8,
                                                      edgecolor='none', zorder=10)
                            ax.add_patch(footer_box)
                            
                            ax.text(0.5, 0.0, 
                                   f'🧠 Generated by Intellexa AI • {datetime.now().strftime("%B %d, %Y")} • Type: {diagram_type.title()}',
                                   ha='center', va='center', fontsize=8,
                                   color='#1565C0', fontweight='bold', zorder=11)
                            
                            # Save with higher quality
                            visual_rep_path = "visual_representation_diagram.png"
                            plt.savefig(visual_rep_path, dpi=300, bbox_inches='tight',
                                       facecolor='#F5F7FA', edgecolor='none')
                            plt.close()
                            
                            # Display the enhanced diagram
                            st.image(visual_rep_path, 
                                    caption=f"🎨 Enhanced Visual Representation: {main_concept}",
                                    use_column_width=True)
                            
                            # Add metrics about the diagram
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            with col_metrics1:
                                st.metric("📊 Diagram Type", diagram_type.title())
                            with col_metrics2:
                                st.metric("🔢 Components", num_elements)
                            with col_metrics3:
                                st.metric("🎯 Accuracy", "AI-Enhanced")
                            
                            # Download option with better styling
                            st.markdown("---")
                            col_dl1, col_dl2 = st.columns([3, 1])
                            
                            with col_dl1:
                                st.info("💡 **Tip:** This diagram was generated based on AI analysis of the visual representation description for maximum accuracy.")
                            
                            with col_dl2:
                                with open(visual_rep_path, "rb") as img_file:
                                    st.download_button(
                                        label="⬇️ Download HD Diagram",
                                        data=img_file,
                                        file_name=f"{doubt.replace(' ', '_')}_visual_rep_HD.png",
                                        mime="image/png",
                                        key="download_visual_rep",
                                        use_container_width=True
                                    )
                            
                            # Show component details
                            with st.expander("📋 View Diagram Components & Labels"):
                                st.markdown("**Component Labels:**")
                                for idx, label in enumerate(element_labels, 1):
                                    st.write(f"{idx}. **{label}**")
                                
                                st.markdown("---")
                                st.markdown("**Connection Type:**")
                                st.info(connections.title())
                                
                                st.markdown("---")
                                st.markdown("**Full Visual Representation Description:**")
                                st.text_area("", visual_rep_content, height=150, key="full_desc")
                            
                            # Show AI specification used
                            with st.expander("🤖 View AI Diagram Specification"):
                                st.markdown("**AI-Generated Specification:**")
                                st.code(diagram_spec, language="text")
                                st.caption("This specification was used to create the accurate visual diagram")
                        
                        else:
                            st.warning("⚠️ Could not extract Visual Representation section from the AI response.")
                        
                    except Exception as viz_error:
                        st.error(f"Visual representation diagram error: {viz_error}")
                        st.info("💡 The AI is still learning to generate accurate diagrams. Try rephrasing your concept.")
                        import traceback
                        with st.expander("🔍 Show Error Details"):
                            st.code(traceback.format_exc())
                            angles = np.linspace(0, 2*np.pi, num_elements, endpoint=False)
                            colors = plt.cm.Greens(np.linspace(0.4, 0.8, num_elements))
                                
                            # Draw circle path
                            circle_angles = np.linspace(0, 2*np.pi, 100)
                            circle_x = center_x + radius * np.cos(circle_angles)
                            circle_y = center_y + radius * np.sin(circle_angles)
                            ax.plot(circle_x, circle_y, 'k--', linewidth=2, alpha=0.3,
                                transform=ax.transAxes)
                                
                            for i, (angle, color) in enumerate(zip(angles, colors)):
                                x = center_x + radius * np.cos(angle)
                                y = center_y + radius * np.sin(angle)
                                    
                                # Draw node
                                node_circle = plt.Circle((x, y), 0.07,
                                                        facecolor=color, edgecolor='#1a1a1a',
                                                        linewidth=2, transform=ax.transAxes,
                                                        zorder=2)
                                ax.add_patch(node_circle)
                                ax.text(x, y, f'{i+1}',
                                        ha='center', va='center', fontsize=11,
                                        fontweight='bold', color='white',
                                        transform=ax.transAxes, zorder=3)
                                    
                                # Draw directional arrows
                                next_angle = angles[(i + 1) % num_elements]
                                next_x = center_x + radius * np.cos(next_angle)
                                next_y = center_y + radius * np.sin(next_angle)
                                    
                                # Calculate arrow midpoint
                                mid_x = (x + next_x) / 2
                                mid_y = (y + next_y) / 2
                                    
                                ax.annotate('', xy=(next_x, next_y), xytext=(x, y),
                                            arrowprops=dict(arrowstyle='->', lw=2.5,
                                                         color='#1a1a1a'),
                                            transform=ax.transAxes, zorder=1)
                            
                            else:
                                # DEFAULT: COMPONENT BLOCKS
                                ax.text(0.5, 0.92, '🔷 Component Structure', 
                                       ha='center', fontsize=14, fontweight='bold',
                                       color='#4A5899', transform=ax.transAxes)
                                
                                # Create grid of components
                                cols = min(3, num_elements)
                                rows = (num_elements + cols - 1) // cols
                                
                                colors = plt.cm.Pastel1(np.linspace(0, 1, num_elements))
                                
                                component_idx = 0
                                for row in range(rows):
                                    for col in range(cols):
                                        if component_idx >= num_elements:
                                            break
                                        
                                        x = 0.15 + col * 0.28
                                        y = 0.65 - row * 0.25
                                        
                                        # Draw component box
                                        rect = plt.Rectangle((x, y), 0.22, 0.15,
                                                            facecolor=colors[component_idx],
                                                            edgecolor='#1a1a1a', linewidth=2,
                                                            transform=ax.transAxes)
                                        ax.add_patch(rect)
                                        
                                        # Add label
                                        ax.text(x + 0.11, y + 0.075, f'C{component_idx + 1}',
                                               ha='center', va='center', fontsize=12,
                                               fontweight='bold', color='#1a1a1a',
                                               transform=ax.transAxes)
                                        
                                        component_idx += 1
                            
                            # Add description box at bottom
                            desc_box = plt.Rectangle((0.05, 0.02), 0.9, 0.1,
                                                    facecolor='white', alpha=0.9,
                                                    edgecolor='#333333', linewidth=2,
                                                    transform=ax.transAxes)
                            ax.add_patch(desc_box)
                            
                            # Add truncated description
                            max_desc_length = 200
                            display_desc = visual_rep_content[:max_desc_length]
                            if len(visual_rep_content) > max_desc_length:
                                display_desc += "..."
                            
                            ax.text(0.5, 0.07, display_desc,
                                   ha='center', va='center', fontsize=9,
                                   wrap=True, color='#333333',
                                   transform=ax.transAxes)
                            
                            # Add footer
                            ax.text(0.5, -0.02, f'Auto-generated by Intellexa AI • {datetime.now().strftime("%Y-%m-%d")}',
                                   ha='center', va='top', fontsize=8,
                                   color='#666666', transform=ax.transAxes)
                            
                            # Save and display
                            visual_rep_path = "visual_representation_diagram.png"
                            plt.savefig(visual_rep_path, dpi=200, bbox_inches='tight',
                                       facecolor='#F0F8FF', edgecolor='none')
                            plt.close()
                            
                            st.image(visual_rep_path, 
                                    caption="🎨 AI-Generated Visual Representation Diagram",
                                    use_column_width=True)
                            
                            # Download option
                            with open(visual_rep_path, "rb") as img_file:
                                st.download_button(
                                    label="⬇️ Download Visual Representation Diagram",
                                    data=img_file,
                                    file_name=f"{doubt.replace(' ', '_')}_visual_rep.png",
                                    mime="image/png",
                                    key="download_visual_rep"
                                )
                            
                            # Show the extracted description
                            with st.expander("📝 View Full Visual Representation Description"):
                                st.info(visual_rep_content)
                        
                    except Exception as viz_error:
                        st.warning(f"Visual representation diagram error: {viz_error}")
                        import traceback
                        st.code(traceback.format_exc())

                    
                    
            
                
                    # Step 3: Generate comprehensive static diagram
                    try:
                        # Parse the explanation into structured sections
                        sections = {
                            'Main Concept': [],
                            'Key Components': [],
                            'How It Works': [],
                            'Visual Representation': [],
                            'Simple Analogy': [],
                            'Practical Application': []
                        }
                        
                        current_section = None
                        lines = explanation.split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Check if line is a section header
                            section_found = False
                            for section_name in sections.keys():
                                if section_name in line and ':' in line:
                                    current_section = section_name
                                    # Add the content after the colon
                                    content = line.split(':', 1)[1].strip()
                                    if content:
                                        sections[section_name].append(content)
                                    section_found = True
                                    break
                            
                            # If not a header and we have a current section, add to that section
                            if not section_found and current_section:
                                sections[current_section].append(line)
                        
                        # Create enhanced visual diagram
                        fig = plt.figure(figsize=(16, 20))
                        fig.patch.set_facecolor('#F8F9FA')
                        
                        # Define colors for each section
                        section_colors = {
                            'Main Concept': '#FF6B6B',
                            'Key Components': '#4ECDC4',
                            'How It Works': '#45B7D1',
                            'Visual Representation': '#FFA07A',
                            'Simple Analogy': '#98D8C8',
                            'Practical Application': '#FFD93D'
                        }
                        
                        # Main title
                        fig.suptitle(f'Visual Explanation: {doubt}', 
                                    fontsize=22, fontweight='bold', y=0.98)
                        
                        # Create 6 subplots (one for each section)
                        y_positions = [0.82, 0.68, 0.54, 0.40, 0.26, 0.12]
                        
                        for idx, (section_name, content_lines) in enumerate(sections.items()):
                            ax = fig.add_axes([0.08, y_positions[idx], 0.84, 0.12])
                            ax.axis('off')
                            
                            color = section_colors[section_name]
                            
                            # Section header with icon
                            icons = {
                                'Main Concept': '💡',
                                'Key Components': '🔷',
                                'How It Works': '⚙️',
                                'Visual Representation': '🎨',
                                'Simple Analogy': '🌟',
                                'Practical Application': '🚀'
                            }
                            
                            icon = icons.get(section_name, '📌')
                            
                            # Draw section header box
                            header_box = plt.Rectangle((0, 0.7), 1, 0.28, 
                                                       facecolor=color, alpha=0.9,
                                                       edgecolor='black', linewidth=2,
                                                       transform=ax.transAxes)
                            ax.add_patch(header_box)
                            
                            # Section title
                            ax.text(0.5, 0.84, f'{icon} {section_name}', 
                                   ha='center', va='center',
                                   fontsize=14, fontweight='bold', color='white',
                                   transform=ax.transAxes)
                            
                            # Content box
                            content_box = plt.Rectangle((0, 0), 1, 0.68,
                                                       facecolor='white', alpha=0.95,
                                                       edgecolor=color, linewidth=2,
                                                       transform=ax.transAxes)
                            ax.add_patch(content_box)
                            
                            # Add content text
                            if content_lines:
                                # Combine all lines and wrap text
                                full_content = ' '.join(content_lines)
                                
                                # Smart text wrapping
                                words = full_content.split()
                                wrapped_lines = []
                                current_line = ''
                                max_chars = 100
                                
                                for word in words:
                                    if len(current_line + ' ' + word) <= max_chars:
                                        current_line += (' ' + word if current_line else word)
                                    else:
                                        if current_line:
                                            wrapped_lines.append(current_line)
                                        current_line = word
                                
                                if current_line:
                                    wrapped_lines.append(current_line)
                                
                                # Display up to 4 lines
                                y_text = 0.55
                                for line_text in wrapped_lines[:4]:
                                    ax.text(0.5, y_text, line_text,
                                           ha='center', va='top',
                                           fontsize=10, wrap=True,
                                           color='#333333',
                                           transform=ax.transAxes)
                                    y_text -= 0.15
                                
                                # If more content exists, add ellipsis
                                if len(wrapped_lines) > 4:
                                    ax.text(0.5, y_text, '...',
                                           ha='center', va='top',
                                           fontsize=10, color='#666666',
                                           transform=ax.transAxes)
                            else:
                                # No content available
                                ax.text(0.5, 0.34, '[Content not available]',
                                       ha='center', va='center',
                                       fontsize=10, color='#999999',
                                       style='italic',
                                       transform=ax.transAxes)
                        
                        # Add footer
                        footer_text = f'Generated by Intellexa AI • {datetime.now().strftime("%Y-%m-%d %H:%M")}'
                        fig.text(0.5, 0.02, footer_text,
                                ha='center', fontsize=9, color='#666666')
                        
                        # Save diagram
                        diagram_path = "visual_explanation.png"
                        plt.savefig(diagram_path, dpi=200, bbox_inches='tight', 
                                   facecolor='#F8F9FA', edgecolor='none')
                        plt.close()
                        
                        st.markdown("### 📊 Complete Visual Diagram")
                        st.image(diagram_path, caption="📊 Comprehensive Concept Breakdown", use_column_width=True)
                        
                        # Download option
                        with open(diagram_path, "rb") as img_file:
                            st.download_button(
                                label="⬇️ Download Visual Diagram",
                                data=img_file,
                                file_name=f"{doubt.replace(' ', '_')}_diagram.png",
                                mime="image/png"
                            )
                    
                    except Exception as viz_error:
                        st.warning(f"Static diagram error: {viz_error}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # Step 4: Enhanced Audio explanation
                    try:
                        st.markdown("### 🔊 Audio Explanation")
                        st.info("📻 Listen to the complete AI explanation with all sections covered")
                        
                        # Parse and structure the explanation for better audio flow
                        audio_sections = []
                        
                        # Add introduction
                        audio_sections.append(f"Let me explain the concept of {doubt}.")
                        
                        # Parse each section of the explanation
                        lines = explanation.split('\n')
                        current_section = ""
                        
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Check if it's a section header
                                if any(header in line for header in ['Main Concept:', 'Key Components:', 'How It Works:', 
                                                                      'Visual Representation:', 'Simple Analogy:', 
                                                                      'Practical Application:']):
                                    if current_section:
                                        audio_sections.append(current_section)
                                    current_section = line.replace(':', '. ')
                                else:
                                    current_section += " " + line
                        
                        # Add the last section
                        if current_section:
                            audio_sections.append(current_section)
                        
                        # Join all sections for complete audio
                        full_audio_text = " ... ".join(audio_sections)
                        
                        # Clean up text for better TTS
                        full_audio_text = full_audio_text.replace('[', '').replace(']', '')
                        full_audio_text = re.sub(r'\d+\.', '', full_audio_text)  # Remove numbering
                        
                        # Add conclusion
                        full_audio_text += f" ... This concludes the explanation of {doubt}."
                        
                        # Display audio controls
                        col_audio1, col_audio2 = st.columns([3, 1])
                        
                        with col_audio2:
                            st.metric("Audio Duration", f"~{len(full_audio_text.split())//150 + 1} min")
                            st.caption("🎧 Use headphones for best experience")
                        
                        with col_audio1:
                            # Generate audio
                            with st.spinner("🎙️ Generating complete audio explanation..."):
                                tts = gTTS(text=full_audio_text, lang="en", slow=False)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                                    audio_path = tmpfile.name
                                    tts.save(audio_path)
                                
                                st.audio(audio_path, format="audio/mp3")
                                
                                # Download option
                                with open(audio_path, "rb") as audio_file:
                                    st.download_button(
                                        label="⬇️ Download Audio Explanation",
                                        data=audio_file,
                                        file_name=f"{doubt.replace(' ', '_')}_explanation.mp3",
                                        mime="audio/mp3"
                                    )
                        
                        # Show transcript
                        with st.expander("📝 View Audio Transcript"):
                            st.text_area("Full Audio Content", full_audio_text, height=200)
                        
                    except Exception as tts_error:
                        st.warning(f"Audio generation error: {tts_error}")
                        st.info("💡 Tip: Try with a shorter concept name or check your internet connection")
                    
                except Exception as e:
                    st.error(f"❌ Error generating explanation: {e}")
    
    # Additional visualization options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔷 Generate Concept Map")
        if st.button("🗺️ Create Interactive Concept Map"):
            if not doubt.strip():
                st.warning("Please enter a concept first.")
            else:
                with st.spinner("Building concept map..."):
                    try:
                        map_prompt = f"""
                        For the concept '{doubt}', provide:
                        1. Central concept name (1-3 words)
                        2. 5-7 related sub-concepts (each 1-3 words)
                        3. Brief relationship between central and each sub-concept
                        
                        Format as:
                        CENTRAL: [main concept]
                        SUB1: [sub-concept] - [relationship]
                        SUB2: [sub-concept] - [relationship]
                        SUB3: [sub-concept] - [relationship]
                        SUB4: [sub-concept] - [relationship]
                        SUB5: [sub-concept] - [relationship]
                        """
                        
                        map_data = call_openrouter(map_prompt, timeout=60)
                        
                        # Parse the response
                        lines = map_data.split('\n')
                        central = "Main Concept"
                        nodes = []
                        
                        for line in lines:
                            line = line.strip()
                            if 'CENTRAL:' in line.upper():
                                try:
                                    central = line.split(':', 1)[1].split('-')[0].strip()
                                    if not central:
                                        central = "Main Concept"
                                except:
                                    central = "Main Concept"
                            elif 'SUB' in line.upper() and ':' in line:
                                try:
                                    parts = line.split(':', 1)[1].split('-')
                                    if parts and parts[0].strip():
                                        nodes.append(parts[0].strip())
                                except:
                                    pass
                        
                        # Ensure we have nodes
                        if not nodes:
                            nodes = ["Component 1", "Component 2", "Component 3", "Component 4", "Component 5"]
                        
                        # Create network graph
                        G = nx.Graph()
                        G.add_node(central)
                        for node in nodes[:7]:
                            if node and node != central:
                                G.add_edge(central, node)
                        
                        # Draw with better layout
                        fig = plt.figure(figsize=(14, 10))
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                        
                        # Draw all nodes first
                        node_list = list(G.nodes())
                        other_nodes = [n for n in node_list if n != central]
                        
                        # Draw peripheral nodes
                        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                                              node_color='#4CAF50', 
                                              node_size=3000, alpha=0.9,
                                              edgecolors='black', linewidths=2)
                        
                        # Draw central node
                        nx.draw_networkx_nodes(G, pos, nodelist=[central], 
                                              node_color='#FF5722', node_size=5000, alpha=0.9,
                                              edgecolors='black', linewidths=3)
                        
                        # Draw edges and labels
                        nx.draw_networkx_edges(G, pos, width=3, alpha=0.6, edge_color='#333333')
                        nx.draw_networkx_labels(G, pos, font_size=11, 
                                               font_weight='bold', font_color='black')
                        
                        plt.title(f"Concept Map: {doubt}", fontsize=18, fontweight='bold', pad=20)
                        plt.axis('off')
                        plt.tight_layout()
                        
                        map_path = "concept_map.png"
                        plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
                        st.image(map_path, caption="🗺️ Interactive Concept Map", use_column_width=True)
                        
                        # Show raw data
                        with st.expander("📋 View Concept Relationships"):
                            st.info(map_data)
                        
                    except Exception as e:
                        st.error(f"❌ Error creating concept map: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.subheader("📊 Generate Mermaid Diagram")
        if st.button("🔷 Create Flow Diagram"):
            if not doubt.strip():
                st.warning("Please enter a concept first.")
            else:
                with st.spinner("Creating diagram..."):
                    try:
                        diagram_prompt = f"""
                        Create a Mermaid flowchart diagram to explain: {doubt}
                        
                        Return ONLY the mermaid code (starting with ```mermaid and ending with ```).
                        Use flowchart syntax (graph TD or graph LR).
                        Keep it simple with 5-10 nodes maximum.
                        Use descriptive labels and clear arrows.
                        
                        Example format:
                        ```mermaid
                        graph TD
                            A[Start] --> B[Process]
                            B --> C[Result]
                        ```
                        """
                        
                        mermaid_code = call_openrouter(diagram_prompt, timeout=60)
                        
                        # Extract mermaid code from markdown
                        mermaid_match = re.search(r'```mermaid\n(.*?)\n```', mermaid_code, re.DOTALL)
                        
                        if mermaid_match:
                            mermaid_diagram = mermaid_match.group(1).strip()
                        else:
                            # Try without code blocks
                            mermaid_match = re.search(r'graph (TD|LR|TB|BT|RL)(.*)', mermaid_code, re.DOTALL)
                            if mermaid_match:
                                mermaid_diagram = f"graph {mermaid_match.group(1)}{mermaid_match.group(2)}"
                            else:
                                mermaid_diagram = mermaid_code
                        
                        st.success("✅ Diagram generated successfully!")
                        
                        # Display mermaid diagram using streamlit components
                        st.markdown("### 🎨 Interactive Mermaid Diagram")
                        
                        # Create HTML with mermaid rendering
                        mermaid_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                            <script>
                                mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
                            </script>
                            <style>
                                body {{
                                    font-family: Arial, sans-serif;
                                    padding: 20px;
                                    background-color: #f5f5f5;
                                }}
                                .mermaid {{
                                    background-color: white;
                                    padding: 20px;
                                    border-radius: 10px;
                                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                                }}
                            </style>
                        </head>
                        <body>
                            <div class="mermaid">
                                {mermaid_diagram}
                            </div>
                        </body>
                        </html>
                        """
                        
                        # Display using streamlit components
                        import streamlit.components.v1 as components
                        components.html(mermaid_html, height=600, scrolling=True)
                        
                        # Show code in expander
                        with st.expander("📋 View Mermaid Code"):
                            st.code(mermaid_diagram, language="mermaid")
                            st.info("💡 You can copy this code and paste it into https://mermaid.live to edit or download")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating diagram: {e}")
                        import traceback
                        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #666;'>🧠 <b>Intellexa AI Tutor</b> — Powered by Advanced AI & Real Voice/Video Analysis</p>
    <p style='color: #888; font-size: 12px;'>Your personalized learning companion with AI-powered voice and video interview analysis</p>
</div>
""", unsafe_allow_html=True)