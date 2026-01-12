# 📖 Smart Cultural Storyteller

**Smart Cultural Storyteller** is a research-grade, interactive AI storytelling application that dynamically adapts to cultural themes, user choices, and even **facial emotions** in real-time.

It moves beyond simple text generation by integrating **Dynamic Identity Generation**, **Cinematography**, **Moral Evaluation**, and **Privacy-Safe Emotion Detection** to create a deeply immersive experience.

---

## 🚀 Key Features

### 1. 🎭 Dynamic Character Identity (LLM-Based)

- **What it does**: Automatically generates culturally authentic protagonist names, backgrounds, and traits based on *any* theme you enter.
- **How**: Uses `ChatGroq` (Llama 3) to "mug up" and synthesize anthropological knowledge on the fly. No hardcoded lists or static data files.
- **Example**: Input "Inuit Legend" -> Protagonist "Amka", Culture "Arctic Indigenous", Traits "Resilient".

### 2. 😊 Facial Emotion Detection (MediaPipe)

- **What it does**: Reads your facial expression (Happy, Sad, Angry, Surprise, Fear) via webcam to influence the story's tone.
- **Tech**: Built on **MediaPipe Tasks API** (`FaceLandmarker`) with **Blendshapes** for high-precision detection.
- **Privacy**: Runs completely **offline/locally** on your CPU. No images are ever saved or sent to the cloud.
- **Streaming**: Uses a background thread to sample your emotion every 1.0s without lagging the UI.

### 3. 🧠 Pure Dynamic Culture Engine

- **What it does**: Acts as an infinite cultural encyclopedia. Instead of relying on a limited static database, it prompts the LLM to generate a "Knowledge Block" of authentic terms, festivals, and myths for your specific theme.
- **Effect**: Reduces hallucination by grounding the story in a generated "truth" about the culture.

### 4. ⚖️ Moral & Karma Engine

- **What it does**: Analyzes your choices for ethical alignment (Compassion, Courage, Greed).
- **Effect**: Your "Karma Score" shifts the narrative. Too much Greed might lead to a tragic ending, while Courage unlocks heroic paths.

### 5. 🎬 Cinematography & Media

- **What it does**: Generates descriptive visual prompts (Camera Angle, Lighting, Color Palette) based on the story's emotional beat.
- **Output**: Can be connected to image generation APIs (currently configured for rapid prototyping or placeholder/text descriptions).

---

## 🛠️ Tools & Technologies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM Core** | **Groq API** (Llama 3 70B/8B) | Ultra-fast text generation for story, culture, and identity. |
| **Vision AI** | **MediaPipe** (Google) | Real-time, on-device facial landmark and emotion detection. |
| **Interface** | **Gradio** | Interactive web UI with streaming webcam support. |
| **Structure** | **LangChain** | Orchestrating prompts and LLM chains. |
| **Validation** | **Pydantic** | Ensuring strict JSON output formats for reliable parsing. |
| **Language** | **Python 3.11** | Optimized environment for library compatibility. |

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.11 (Recommended)
- A Groq API Key (in `.env`)
- A Google API Key (optional, for other services)

### 1. Clone & Install

```bash
git clone https://github.com/harshith051104/storyteller.git
cd storyteller
```

### 2. Set up Virtual Environment

```bash
# Create environment
python -m venv .venv_311

# Activate (Windows)
.venv_311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=...
```

### 4. Download Model

The app requires the MediaPipe model file. A helper script is included:

```bash
python download_model.py
```

*(This downloads `face_landmarker.task` to the root folder)*

---

## 🎮 How to Run

For convenience, use the included batch script:

```bat
run_storyteller.bat
```

Or run manually:

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`.

---

## 📂 Project Structure

- `app.py`: Main Gradio application and event handlers.
- `story_engine.py`: Core narrative logic, prompt engineering, and LLM interaction.
- `emotion_engine.py`: MediaPipe integration for facial expression recognition.
- `character_engine.py`: Handles dynamic identity generation.
- `culture_engine.py`: Generates on-demand cultural context.
- `moral_engine.py`: Evaluates user choices and tracks karma.
- `media_engine.py`: Manages image/audio prompt generation.
- `config.py`: Configuration constants.

