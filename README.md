# 🧠 Intellexa AI Tutor

**Your Ultimate AI-Powered Learning & Interview Coach**

Intellexa is a comprehensive AI learning platform that combines personalized education, adaptive tutoring, and advanced interview preparation with voice and video analysis capabilities.

---

## ✨ Features

### 📚 1. AI-Powered Learning Plan Generator
- **Personalized Study Plans**: Generate custom learning roadmaps based on your goals, skill level, and time availability
- **Progress Tracking**: Mark days as completed and track your learning journey
- **PDF Export**: Download your complete learning plan for offline reference
- **Adaptive Scheduling**: Plans tailored to 5-30 days with 1-6 hours daily study time
- **Multiple Domains**: Web Development, Data Science, Machine Learning, MERN Stack, Java, Android, and more

### 🤖 2. Adaptive AI Tutor Assistant
- **Contextual Learning**: AI tutor that adapts to your skill level (Beginner/Intermediate/Advanced)
- **Interactive Actions**:
  - Clear topic explanations with analogies
  - Practice question generation
  - Study improvement tips
  - Next topic suggestions
- **Personalized Teaching Style**: Customized explanations based on your learning level

### 🎤 3. AI Interview Coach (Multi-Modal)

#### 📝 Text-Based Interview Practice
- Generate up to 150 interview questions per domain
- Instant AI feedback on your written answers
- Scoring on Technical Accuracy, Clarity, Confidence, and Professionalism
- Domain-specific questions (General, Cybersecurity, AI/ML, Web Dev, Data Science)

#### 🎙️ Voice Interview Analysis (Advanced)
- **Speech Transcription**: Automatic conversion of audio to text
- **Voice Metrics Analysis**:
  - Speech pace detection (Fast/Moderate/Slow)
  - Pause pattern analysis
  - Filler words detection (um, uh, like, etc.)
  - Speaking time vs pause time breakdown
  - Clarity score (0-10)
- **Comprehensive Scoring**:
  - Content Analysis (Technical accuracy)
  - Verbal Communication (Speech quality)
  - Fluency & Coherence
  - Confidence & Professionalism
  - Overall Impression
- **Visual Analytics**: Performance radar charts, time distribution pie charts
- **Downloadable Reports**: Complete analysis in text format

#### 🎥 Video Interview Analysis (Full AI Analysis)
- **Computer Vision Analysis**:
  - Real-time emotion detection (Happy, Neutral, Sad, Angry, Surprise, Fear)
  - Eye contact tracking with percentage metrics
  - Gesture and posture analysis (Minimal/Moderate/Excessive)
  - Facial expression changes
- **Speech Pattern Recognition**: Full audio analysis from video
- **Multi-Dimensional Scoring**:
  - Content Analysis
  - Verbal Communication
  - Non-Verbal Communication
  - Emotional Intelligence
  - Overall Interview Readiness
- **Progress Tracking**: Performance improvement across multiple questions
- **Detailed Reports**: Comprehensive video analysis with all metrics

### 📊 4. Progress Dashboard
- **Learning Plan Progress**: Visual progress bars for completed study days
- **Interview History**: Complete record of all text, voice, and video interviews
- **Score Trends**: Line charts showing improvement over time
- **Performance Comparison**: Bar charts comparing scores across questions
- **Achievement Badges**: 
  - 🥇 First Interview badges (Text/Voice/Video)
  - 💡 Confidence Master
  - 🧠 Technical Expert
  - 🎤 Voice Interview Pro
  - ⭐ Video Interview Pro
  - 🏆 Complete Interview Champion

### ⚡ 5. AI Doubt Visualizer

#### 🎨 Enhanced Visual Representation
- **AI-Powered Diagram Generation**: Automatically creates accurate diagrams based on concept descriptions
- **5 Intelligent Diagram Types**:
  - 🔄 **Flowchart**: Process flows and sequential steps
  - 🕸️ **Network**: Hub-spoke architectures and connections
  - 📊 **Hierarchical**: Layered structures and pyramids
  - 🔁 **Circular/Cycle**: Iterative processes and loops
  - 🔷 **Component Blocks**: Modular architectures
- **Smart Label Extraction**: Uses AI to identify and label diagram components accurately
- **High-Quality Export**: 300 DPI diagrams with professional styling
- **Component Details**: Expandable view showing all labels and connections

#### 📊 Complete Visual Explanation
- **6-Section Breakdown**:
  - 💡 Main Concept
  - 🔷 Key Components
  - ⚙️ How It Works
  - 🎨 Visual Representation
  - 🌟 Simple Analogy
  - 🚀 Practical Application
- **Comprehensive Diagram**: All sections in one visual infographic
- **Color-Coded Sections**: Easy-to-distinguish information blocks

#### 🔊 Audio Explanations
- **Text-to-Speech**: Complete concept explanation in audio format
- **Structured Narration**: Section-by-section audio breakdown
- **Download Support**: MP3 export for offline learning
- **Transcript View**: Read along with audio

#### 🗺️ Concept Mapping
- **Interactive Network Graphs**: Visual concept relationships
- **Central-Node Architecture**: Main concept with related sub-concepts
- **NetworkX Powered**: Professional graph visualization

#### 🔷 Mermaid Diagrams
- **Interactive Flowcharts**: AI-generated Mermaid diagrams
- **Live Rendering**: In-browser diagram visualization
- **Code Export**: Copy diagram code for external use

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/Pabitra3/Advanced_Ai.git
cd Advanced_Ai
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
IMAGEGEN_API_KEY=your_imagegen_api_key_here
```

**Get Your API Keys:**
- OpenRouter API: [https://openrouter.ai](https://openrouter.ai)
- ImageGen API (optional): [Your provider]

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📦 Requirements

```txt
streamlit>=1.28.0
requests>=2.31.0
python-dotenv>=1.0.0
gtts>=2.4.0
matplotlib>=3.7.0
pandas>=2.0.0
networkx>=3.1
fpdf>=1.7.2
numpy>=1.24.0
opencv-python>=4.8.0
fer>=22.5.0
mediapipe>=0.10.0
SpeechRecognition>=3.10.0
moviepy>=1.0.3
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.1
imageio>=2.31.0
```

---

## 🎯 Usage Guide

### 1. Setting Up Your Profile
1. Open the **sidebar** on the left
2. Enter your **name**
3. Select your **skill level** (Beginner/Intermediate/Advanced)
4. Choose your **daily study time** (1-6 hours)
5. Set **completion target** (5-30 days)

### 2. Generating a Learning Plan
1. Go to the **📚 Learning Plan** tab
2. Select your **goal** from the dropdown
3. Click **✨ Generate AI Learning Plan**
4. Review your personalized plan
5. Mark days as completed using checkboxes
6. Download as PDF for offline access

### 3. Using the AI Tutor
1. Navigate to **🤖 AI Tutor** tab
2. Select a topic from your learning plan
3. Choose an action:
   - Explain Topic Clearly
   - Generate Practice Questions
   - Suggest Next Topic
   - Give Study Improvement Tips
4. Click **Ask AI ✨**
5. Review the AI-generated response

### 4. Interview Practice

#### Text Interview:
1. Go to **🎤 AI Interview Coach** tab
2. Select **📝 Text-Based Interview**
3. Enter number of questions (1-150)
4. Choose your domain
5. Click **🧩 Generate Interview Questions**
6. Type your answers
7. Click **Evaluate Answer** for instant feedback

#### Voice Interview:
1. Select **🎙️ Voice Interview (Audio Analysis)**
2. Generate questions
3. Record your audio response (WAV, MP3, M4A, OGG, FLAC)
4. Upload the recording
5. Click **🔍 Analyze Voice Interview**
6. Review comprehensive voice metrics
7. Download detailed report

#### Video Interview:
1. Select **🎥 Video Interview (Full Analysis)**
2. Generate questions
3. Record yourself answering (MP4, AVI, MOV, WEBM)
4. Upload your video
5. Click **🔍 Analyze Video Interview**
6. Get AI analysis of emotions, eye contact, gestures, and speech
7. Download comprehensive report

### 5. Visualizing Concepts
1. Open **⚡ AI Doubt Visualizer** tab
2. Enter your concept/doubt (e.g., "Neural Networks")
3. Click **🎨 Generate Visual Explanation**
4. Get:
   - Complete text explanation
   - Enhanced visual representation diagram
   - Comprehensive 6-section visual breakdown
   - Audio explanation
   - Component details
5. Optional tools:
   - **🗺️ Create Interactive Concept Map**
   - **🔷 Create Flow Diagram** (Mermaid)

### 6. Tracking Progress
1. Visit **📊 Progress Dashboard**
2. View:
   - Learning plan completion percentage
   - Interview history (Text/Voice/Video)
   - Score trends over time
   - Performance comparisons
   - Achievement badges
3. Monitor improvement across all interview types

---

## 🛠️ Technologies Used

### Frontend & Framework
- **Streamlit**: Interactive web application framework
- **Matplotlib**: Data visualization and chart generation
- **Plotly**: Interactive 3D visualizations

### AI & Machine Learning
- **OpenRouter API**: GPT-4 powered AI responses
- **FER (Facial Expression Recognition)**: Emotion detection
- **MediaPipe**: Face mesh and pose estimation
- **OpenCV**: Computer vision processing
- **Librosa**: Audio analysis and feature extraction

### Audio/Video Processing
- **gTTS (Google Text-to-Speech)**: Audio generation
- **SpeechRecognition**: Audio transcription
- **MoviePy**: Video processing and audio extraction
- **PyDub**: Audio format conversion

### Data & Visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **NetworkX**: Graph theory and concept mapping
- **ImageIO**: GIF creation for animations

### Document Generation
- **FPDF**: PDF generation for learning plans
- **Markdown**: Report formatting

---

## 📊 Project Structure

```
intellexa-ai-tutor/
│
├── app.py                          # Main application file
├── .env                            # Environment variables (API keys)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── temp_files/                     # Temporary storage (auto-generated)
│   ├── audio_files/
│   ├── video_files/
│   └── diagrams/
│
└── downloads/                      # User downloads (auto-generated)
    ├── learning_plans/
    ├── interview_reports/
    └── visualizations/
```

---

## 🎨 Key Features Breakdown

### Voice Analysis Metrics
- **Transcription**: Google Speech Recognition API
- **Speech Rate**: Calculated as speaking_time / total_duration
- **Pause Detection**: RMS energy threshold analysis
- **Filler Words**: Pattern matching for common fillers
- **Clarity Score**: Composite score based on pace, pauses, and fillers

### Video Analysis Metrics
- **Emotion Detection**: FER with MTCNN face detection
- **Eye Contact**: Face mesh landmark analysis (eye alignment)
- **Gestures**: Pose estimation tracking hand movements relative to shoulders
- **Frame Sampling**: Analyzes every 5th frame for performance optimization

### Diagram Generation Logic
1. **AI Analysis**: Extracts visual representation from explanation
2. **Specification Generation**: Creates structured diagram requirements
3. **Type Detection**: Identifies appropriate diagram type from keywords
4. **Label Extraction**: Parses component names from AI response
5. **Rendering**: Matplotlib-based professional diagram creation
6. **Post-Processing**: High-DPI export with shadows and styling

---

## 🔒 Privacy & Security

- **Local Processing**: All audio/video analysis happens locally
- **Temporary Files**: Automatically cleaned after processing
- **API Security**: Keys stored in .env file (never committed to Git)
- **No Data Storage**: Interview data stored only in session state
- **User Control**: All data can be downloaded and deleted

---

## 🐛 Troubleshooting

### Common Issues

#### API Key Errors
```
Error: OPENROUTER_API_KEY not found
```
**Solution**: Create a `.env` file with your API key

#### Audio Transcription Fails
```
Transcription warning: ...
```
**Solution**: 
- Ensure audio is clear with minimal background noise
- Use supported formats (WAV, MP3, M4A)
- Check microphone settings

#### Video Analysis Slow
```
Analyzing frame X/Y...
```
**Solution**: This is normal. Video analysis takes 1-2 minutes depending on video length

#### Module Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Solution**: Run `pip install -r requirements.txt` again

#### Diagram Generation Fails
```
Visual representation diagram error
```
**Solution**: 
- Check internet connection (API calls required)
- Try rephrasing your concept
- Ensure matplotlib is properly installed

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Test all features before submitting
- Update README for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** for providing AI API access
- **Streamlit** for the amazing web framework
- **FER** for facial emotion recognition
- **MediaPipe** for pose and face mesh detection
- **Google** for Text-to-Speech and Speech Recognition
- **Open Source Community** for all the incredible libraries

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Pabitra3/Advanced_Ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pabitra3/Advanced_Ai/discussions)
- **Email**: developersivaay@gmail.com

---

## 🗺️ Roadmap

### Upcoming Features
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Real-time video interview practice
- [ ] Group study sessions
- [ ] Gamification with XP and levels
- [ ] Integration with more AI models
- [ ] Custom diagram templates
- [ ] Advanced analytics dashboard
- [ ] Social learning features
- [ ] API for third-party integrations

---

## 📈 Version History

### v1.0.0 (Current)
- ✅ AI Learning Plan Generator
- ✅ Adaptive AI Tutor
- ✅ Text Interview Coach
- ✅ Voice Interview Analysis
- ✅ Video Interview Analysis
- ✅ Enhanced Visual Doubt Visualizer
- ✅ Progress Dashboard
- ✅ Achievement System

---

## 💡 Tips for Best Results

### Voice Interviews
- 🎤 Use a good quality microphone
- 🔇 Record in a quiet environment
- 🗣️ Speak clearly and at moderate pace
- ⏱️ Answer comprehensively (aim for 30-90 seconds)

### Video Interviews
- 📹 Ensure good lighting on your face
- 👁️ Look at the camera (simulates eye contact)
- 🪑 Sit upright with good posture
- 👔 Dress professionally
- 📱 Use stable camera position

### Learning Plans
- ✅ Mark days as complete honestly
- 📚 Follow the plan consistently
- 🔄 Adjust hours based on actual study time
- 📝 Use the AI Tutor for difficult topics

### Concept Visualization
- 📝 Be specific with your doubt/concept
- 🎨 Try different phrasing if diagram isn't accurate
- 🔊 Use audio explanation while studying diagrams
- 💾 Download diagrams for future reference

---

<div align="center">

## 🌟 Star This Repo!

If you find Intellexa helpful, please give it a ⭐️ on GitHub!

**Made with ❤️ by Pabitra Kumar Sahoo**

[Report Bug](https://github.com/Pabitra3/Advanced_Ai/issues) · [Request Feature](https://github.com/Pabitra3/Advanced_Ai/issues) · [Documentation](https://github.com/Pabitra3/Advanced_Ai/wiki)

</div>

---

**Happy Learning! 🚀📚🎓**