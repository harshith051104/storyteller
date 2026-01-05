import gradio as gr
import os
from dotenv import load_dotenv
from story_engine import StoryTeller
from media_engine import MediaEngine

# Load environment variables
load_dotenv()

# Initialize Engines
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file with your GOOGLE_API_KEY. See USAGE_GUIDE.md for details.")
    exit(1)

try:
    story_teller = StoryTeller()
    media_engine = MediaEngine()
except Exception as e:
    print(f"Failed to initialize engines: {e}")
    exit(1)

def start_story_handler(theme):
    if not theme:
        return "Please enter a theme to start.", None, None
    
    story_text = story_teller.start_story(theme)
    # Generate Media
    audio_path = media_engine.generate_audio(story_text)
    media_path, media_type = media_engine.generate_scene(story_text)
    
    # Return updates for Gradio components: (Story, Audio, Video, Image)
    if media_type == "video":
        return story_text, audio_path, gr.update(value=media_path, visible=True), gr.update(visible=False)
    else:
        return story_text, audio_path, gr.update(visible=False), gr.update(value=media_path, visible=True)

def continue_story_handler(user_choice):
    if not user_choice:
        # Return format: story, audio, video_update, image_update
        return "Please make a choice to continue.", None, None, None

    story_text = story_teller.continue_story(user_choice)
    
    # Generate Media
    audio_path = media_engine.generate_audio(story_text)
    media_path, media_type = media_engine.generate_scene(story_text)
    
    if media_type == "video":
        return story_text, audio_path, gr.update(value=media_path, visible=True), gr.update(visible=False)
    else:
        return story_text, audio_path, gr.update(visible=False), gr.update(value=media_path, visible=True)

# Gradio Interface
with gr.Blocks(title="Smart Cultural Storyteller") as demo:
    gr.Markdown("# 📖 Smart Cultural Storyteller")
    gr.Markdown("Experience cultural narratives brought to life with AI.")
    
    with gr.Row():
        with gr.Column(scale=1):
            theme_input = gr.Textbox(label="Enter a Culture or Theme", placeholder="Japanese Folklore...")
            start_btn = gr.Button("Start New Story", variant="primary")
            
            gr.Markdown("---")
            
            choice_input = gr.Textbox(label="Your Choice / Action")
            continue_btn = gr.Button("Make Choice")
            
        with gr.Column(scale=2):
            story_display = gr.Markdown(label="Story")
            with gr.Row():
                # Both components exist but visibility is toggled
                video_display = gr.Video(label="Scene (Video)", height=400, visible=False)
                image_display = gr.Image(label="Scene (Image)", type="filepath", visible=False)
                audio_display = gr.Audio(label="Narration", type="filepath")

    # Event Handlers
    start_btn.click(
        fn=start_story_handler,
        inputs=[theme_input],
        outputs=[story_display, audio_display, video_display, image_display]
    )
    
    continue_btn.click(
        fn=continue_story_handler,
        inputs=[choice_input],
        outputs=[story_display, audio_display, video_display, image_display]
    )

if __name__ == "__main__":
    demo.launch()
