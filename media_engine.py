import os
import time
import traceback
import requests
import base64
import pyttsx3
from cinematography_engine import CinematographyEngine

class MediaEngine:
    def __init__(self):
        # Hugging Face token check
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.hf_token:
            print("WARNING: HUGGINGFACE_API_TOKEN not found. Image generation disabled.")

        if not self.hf_token:
            print("WARNING: HUGGINGFACE_API_TOKEN not found. Image generation disabled.")

        # pyttsx3 is initialized per-call to avoid threading issues on Windows/Gradio




        # Initialize Cinematography Engine
        try:
            self.cine_engine = CinematographyEngine()
        except Exception as e:
            print(f"Failed to init CinematographyEngine: {e}")
            self.cine_engine = None

    # ---------------- IMAGE GENERATION ----------------
    def generate_scene(self, story_text, emotion="neutral", character_desc="", visual_keywords_bypass=None):
        if not self.hf_token:
            return None, "image"

        try:
            # 1. Get Cinematography Keywords
            if visual_keywords_bypass:
                keywords = visual_keywords_bypass
            elif self.cine_engine:
                keywords = self.cine_engine.enhance_prompt(story_text[:500], emotion)
            else:
                keywords = f"cinematic, {emotion} atmosphere"

            # 2. Construct Prompt with Character Consistency
            prompt = f"""
            masterpiece, ultra-detailed cinematic illustration, storybook fantasy art,
            authentic cultural aesthetics, rich textures, 8k resolution,
            {keywords},
            
            SCENE:
            {story_text[:400]}
            
            CHARACTER FOCUS:
            {character_desc}
            
            STYLE:
            digital painting, concept art, unreal engine quality, artstation trending
            
            NEGATIVE:
            blurry, low resolution, distorted face, extra limbs, bad anatomy, watermark, text
            """

            API_URL = (
                "https://router.huggingface.co/hf-inference/models/"
                "black-forest-labs/FLUX.1-schnell"
            )
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"HF Error {response.status_code}: {response.text}")

            filename = f"scene_{int(time.time())}.png"
            output_path = os.path.abspath(filename)

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"SUCCESS: Image saved → {output_path}")
            return output_path, "image"

        except Exception as e:
            error_msg = (
                f"Image generation failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            print(error_msg)

            with open("debug_error.log", "w") as f:
                f.write(error_msg)

            return None, "image"

    # ---------------- AUDIO GENERATION ----------------
    def generate_audio(self, text, voice_id=None):
        try:
            # Re-initialize pyttsx3 per call for thread safety in Gradio
            engine = pyttsx3.init()
            # Optional: Set properties
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            
            output_path = f"audio_{int(time.time())}.mp3"
            abs_output_path = os.path.abspath(output_path)
            
            # Saving to file
            engine.save_to_file(text, abs_output_path)
            engine.runAndWait()
            
            # Explicit cleanup if possible (pyttsx3 doesn't have a close(), but letting it go out of scope helps)
            del engine

            return abs_output_path

        except Exception as e:
            print(f"Audio generation failed: {e}")
            return None

    # ---------------- VIDEO GENERATION ----------------

