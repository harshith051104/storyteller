from config import MODEL_CREATIVE, MAX_HISTORY_TURNS
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from culture_engine import CultureEngine
from logger_config import get_logger

logger = get_logger()

class StoryOutput(BaseModel):
    story_text: str = Field(description="The narrative content (100-150 words) with choices at the end")
    emotion: str = Field(description="One word emotion: joy, sadness, anger, fear, peace, mystery")
    visual_keywords: str = Field(description="Comma-separated visual keywords(Camera Angle, Lighting, Color Palette)")

class StoryTeller:
    def __init__(self):
        self.llm = ChatGroq(model=MODEL_CREATIVE)
        self.history = []
        self.culture_engine = CultureEngine()
        self.language_instruction = "Narrate in English."
        self.parser = JsonOutputParser(pydantic_object=StoryOutput)

    def set_language(self, language="English"):
        if language and language.lower() != "english":
            self.language_instruction = (
                f"Narrate primarily in English, BUT you MUST adhere to the following code-switching rules:\n"
                f"1. Use {language} for ALL opening greetings and significant cultural terms.\n"
                f"2. Quotes and dialogue MUST be in {language} (provide English translation in parentheses if long).\n"
                f"3. Ensure the tone reflects the linguistic nuance of {language}.\n"
                f"Example: 'Namaste! (Hello!) The wind howled...' "
            )
        else:
            self.language_instruction = "Narrate in English."

    def _trim_history(self):
        """Trims history to keep only the most recent interactions."""
        try:
            # Keep system prompt (index 0) + last N turns
            if len(self.history) > MAX_HISTORY_TURNS:
                # Safely slice the last (N-1) elements
                keep_count = MAX_HISTORY_TURNS - 1
                recent_history = self.history[-keep_count:]
                self.history = [self.history[0]] + recent_history
        except Exception as e:
            logger.warning(f"History Trimming Error: {e}")
            # Fallback: Just keep explicit last 5
            if len(self.history) > 5:
                self.history = [self.history[0]] + self.history[-5:]

    def start_story(self, theme, language="English"):
        """Initializes the story based on a theme and cultural context."""
        self.set_language(language)
        
        # 1. Retrieve Cultural Context (RAG)
        logger.info(f"Retrieving cultural context for: {theme}")
        context_str = self.culture_engine.get_context_string(theme)
        
        if context_str:
            grounding_instruction = (
                "You have access to the following trusted cultural knowledge:\n"
                f"{context_str}\n\n"
                "CRITICAL INSTRUCTION: You MUST ground your story in this provided context. "
                "Use specific symbols, names, and festivals mentioned. "
                "Do NOT hallunicate details if they contradict this context."
            )
        else:
            grounding_instruction = "No specific cultural documents found. Rely on general knowledge but remain respectful and authentic."

        # 2. Build System Prompt
        system_prompt = (
            "You are a 'Smart Cultural Storyteller'. Your goal is to preserve and retell cultural narratives "
            "in an engaging, interactive 'choose-your-own-adventure' style.\n\n"
            f"{grounding_instruction}\n\n"
            f"Language Rule: {self.language_instruction}\n\n"
            "Format:\n"
            "- Keep responses concise (100-150 words).\n"
            "- End with exactly 2 or 3 distinct choices.\n"
            "- CRITICAL: Format choices as numbered list: 1. [Choice A] 2. [Choice B] etc.\n"
            "- If information is unknown, acknowledge it subtly or steer towards known elements.\n"
            "OUTPUT JSON ONLY: Return a valid JSON object with keys: 'story_text', 'emotion', 'visual_keywords'."
        )

        self.history = [SystemMessage(content=system_prompt)]
        
        prompt = f"Start a story about {theme}. Set the scene and offer numbered choices.\n{self.parser.get_format_instructions()}"
        self.history.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(self.history)
            self.history.append(response)
            parsed_response = self.parser.parse(response.content)
            return parsed_response
        except Exception as e:
            logger.error(f"Story Start Error: {e}")
            # Fallback
            return {
                "story_text": f"The story begins with {theme}. (Error generating full story)",
                "emotion": "mystery",
                "visual_keywords": "foggy, ancient, mysterious"
            }

    def continue_story(self, user_choice):
        """Continues the story based on user's choice."""
        self._trim_history()
        self.history.append(HumanMessage(content=user_choice))
        
        try:
            response = self.llm.invoke(self.history)
            self.history.append(response)
            parsed_response = self.parser.parse(response.content)
            return parsed_response
        except Exception as e:
            logger.error(f"Story Continue Error: {e}")
            return {
                "story_text": "The story continues... (Error generating segment)",
                "emotion": "neutral",
                "visual_keywords": "standard scene"
            }
