import gradio as gr
import time
import os
from dotenv import load_dotenv
from story_engine import StoryTeller
from media_engine import MediaEngine
from character_engine import CharacterEngine, Character
from moral_engine import MoralEngine
from emotion_engine import EmotionEngine

# Load environment variables
load_dotenv()

# Initialize Engines
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found.")
    exit(1)

story_teller = StoryTeller()
media_engine = MediaEngine()
character_engine = CharacterEngine()
try:
    emotion_engine = EmotionEngine()
except Exception as e:
    print(f"Warning: Emotion Engine failed to load ({e}). Face detection disabled.")
    emotion_engine = None

def start_story_handler(theme, language, history_state):
    try:
        if not theme:
            yield "Please enter a theme.", None, None, history_state, "", ""
            return
        
        # Initialize Character & Moral Engine
        char_name = f"Protagonist_{theme.split()[0]}"
        character = character_engine.initialize_character(char_name, theme)
        moral = MoralEngine()

        # Start Story (Returns JSON dict)
        story_data = story_teller.start_story(theme, language)
        story_text = story_data.get("story_text", "")
        emotion = story_data.get("emotion", "neutral")
        visual_keywords = story_data.get("visual_keywords")
        
        # Immediate yield: Story Text
        yield story_text, None, None, {
            "character": character.to_dict(), 
            "moral_scores": moral.scores
        }, "Compassion: 0 | Courage: 0 | Greed: 0", ""

        # Generate Audio
        audio = media_engine.generate_audio(story_text)
        yield story_text, audio, None, {
            "character": character.to_dict(), 
            "moral_scores": moral.scores
        }, "Compassion: 0 | Courage: 0 | Greed: 0", ""

        # Generate Image (Optimized)
        char_desc = character_engine.get_visual_description(character)
        media_path, media_type = media_engine.generate_scene(story_text, emotion, char_desc, visual_keywords_bypass=visual_keywords)
        
        image_update = gr.update(value=media_path, visible=True) if media_type == "image" else gr.update(visible=False)

        yield (
            story_text, 
            audio, 
            image_update,
            {
                "character": character.to_dict(), 
                "moral_scores": moral.scores
            },
            f"Compassion: 0 | Courage: 0 | Greed: 0",
            "" # Status msg
        )

    except Exception as e:
        print(f"Error: {e}")
        yield f"Error: {e}", None, None, None, "", ""


def process_emotion_stream(image, last_time):
    """
    Background process to detect emotion from stream.
    Throttled to run every 1.0s to avoid log spam/CPU load.
    Returns: (new_emotion_label, new_timestamp)
    """
    current_time = time.time()
    
    # Throttle: Only process if 1.0s passed
    if image is None or (current_time - last_time < 1.0):
        return gr.skip(), gr.skip()
        
    if emotion_engine:
        try:
            # This calling detect_emotion will trigger the print log if confidence > 0.3
            result = emotion_engine.detect_emotion(image)
            label = result["emotion"] if result else "neutral"
            return label, current_time
        except Exception:
            return "neutral", current_time
            
    return "neutral", current_time


def continue_story_handler(user_choice, user_emotion_label, state):
    try:
        if not user_choice:
            yield "Please make a choice.", None, None, state, "", ""
            return

        # Rehydrate State
        if not state:
            yield "Session expired. Start over.", None, None, None, "", ""
            return
        
        character = Character.from_dict(state["character"])
        moral = MoralEngine()
        moral.scores = state["moral_scores"]

        # 1. Score the Choice
        moral_result = moral.score_choice(user_choice, story_teller.history[-1].content)
        
        # 2. Update Character Traits
        character = character_engine.update_traits_from_scores(character, moral.scores)

        # 3. Continue Story (Returns JSON)
        # Use the passed-in emotion label (from State)
        
        # Inject into context
        context_choice = f"{user_choice} (User Facial Emotion: {user_emotion_label})"
        
        story_data = story_teller.continue_story(context_choice)
        story_text = story_data.get("story_text", "")
        # Blend Emotions: Story > Facial
        story_emotion = story_data.get("emotion", "neutral")
        final_emotion = story_emotion if story_emotion != "neutral" else user_emotion_label
        visual_keywords = story_data.get("visual_keywords")
        
        moral_display = f"Compassion: {moral.scores['compassion']} | Courage: {moral.scores['courage']} | Greed: {moral.scores['greed']}"
        
        status_msg = f"✨ Karma Updated! (Compassion: {moral_result.get('compassion')}, Courage: {moral_result.get('courage')}, Greed: {moral_result.get('greed')}) | Face: {user_emotion_label}"

        # Yield Text immediately
        yield story_text, None, None, state, moral_display, status_msg
        
        # 4. Generate Audio
        audio = media_engine.generate_audio(story_text)
        yield story_text, audio, None, state, moral_display, status_msg

        # 5. Generate Media
        char_desc = character_engine.get_visual_description(character)
        media_path, media_type = media_engine.generate_scene(story_text, final_emotion, char_desc, visual_keywords_bypass=visual_keywords)
        image_update = gr.update(value=media_path, visible=True) if media_type == "image" else gr.update(visible=False)

        # Update State
        state["moral_scores"] = moral.scores
        state["character"] = character.to_dict()

        # Check if story ended
        if "THE END" in story_text.upper():
            reflection = moral.generate_reflection()
            story_text += f"\n\n✨ **Moral Reflection**: {reflection}"

        yield story_text, audio, image_update, state, moral_display, status_msg

    except Exception as e:
        print(f"Error: {e}")
        yield f"Error: {e}", None, None, state, "", ""


# Gradio Interface
with gr.Blocks(title="Smart Cultural Storyteller 2.0") as demo:
    state = gr.State({})
    # State to hold background emotion detection
    emotion_state = gr.State("neutral")
    # State to hold last processing timestamp for throttling
    timer_state = gr.State(0.0)

    gr.Markdown("# 📖 Smart Cultural Storyteller 2.0")
    gr.Markdown("Research-Grade Interactive Storytelling with RAG, Cinematography, and Moral Agents.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Section
            theme_input = gr.Textbox(label="Culture / Theme", placeholder="Indian Folklore, Samurai Legends...")
            lang_input = gr.Dropdown(["English", "Hindi", "Telugu", "Japanese", "Spanish"], label="Language", value="English")
            # Enable Streaming
            webcam_input = gr.Image(label="Your Emotion (Optional)", sources=["webcam"], type="numpy", visible=True, streaming=True)
            # Oral History Removed
            
            start_btn = gr.Button("Start New Journey", variant="primary")
            
            gr.Markdown("---")
            
            # Game Controls
            choice_input = gr.Textbox(label="Your Choice / Action")
            continue_btn = gr.Button("Make Choice")

            # Stats Display
            gr.Markdown("### ⚖️ Moral Alignment")
            moral_info = gr.Textbox(interactive=False, label="Karma Score")
            status_info = gr.Textbox(interactive=False, label="Recent Updates")
            
        with gr.Column(scale=2):
            story_display = gr.Markdown(label="Story")
            
            with gr.Row():
                image_display = gr.Image(label="Illustration", type="filepath", visible=False)
            
            audio_display = gr.Audio(label="Narration", type="filepath", autoplay=True)

    # Event Handlers
    
    # Streaming Event
    webcam_input.stream(
        fn=process_emotion_stream,
        inputs=[webcam_input, timer_state],
        outputs=[emotion_state, timer_state],
        show_progress=False
    )

    start_btn.click(
        fn=start_story_handler,
        inputs=[theme_input, lang_input, state],
        outputs=[story_display, audio_display, image_display, state, moral_info, status_info]
    )
    
    continue_btn.click(
        fn=continue_story_handler,
        inputs=[choice_input, emotion_state, state],
        outputs=[story_display, audio_display, image_display, state, moral_info, status_info]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
