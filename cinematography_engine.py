from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import MODEL_FAST
from logger_config import get_logger

logger = get_logger()

class CinematographyEngine:
    def __init__(self):
        # We use a specialized instance for visual instruction
        self.llm = ChatGroq(model=MODEL_FAST, temperature=0.7)

    def enhance_prompt(self, story_segment, emotion):
        """
        Generates a visually rich, cinematic description using an LLM.
        Avoids hardcoded mappings.
        """
        system_instruction = (
            "You are an expert Virtual Cinematographer and Art Director. "
            "Your task is to translate a story segment and an emotion into a precise "
            "visual description for AI Image Generation (Stable Diffusion/Flux). "
            "Focus ONLY on: Camera Angle, Lighting, Color Palette, Depth of Field, and Composition. "
            "Do NOT include the story action itself, just the visual style keywords. "
            "Keep it comma-separated and under 40 words."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "Story: {story}\nEmotion: {emotion}\n\nVisual Keywords:")
        ])

        chain = prompt_template | self.llm
        
        try:
            result = chain.invoke({"story": story_segment, "emotion": emotion})
            return result.content.strip()
        except Exception as e:
            logger.error(f"Cinematography Engine Error: {e}")
            # Fallback if LLM fails
            return f"cinematic shot, {emotion} lighting, 8k resolution"
