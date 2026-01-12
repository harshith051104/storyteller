import random

class Character:
    def __init__(self, name, culture, age=20, traits=None, voice_id="21m00Tcm4TlvDq8ikWAM", face_seed=None):
        self.id = f"char_{random.randint(1000, 9999)}"
        self.name = name
        self.culture = culture
        self.age = age
        self.traits = traits if traits else []
        self.voice_id = voice_id
        # Persistent seed for image generation consistency
        self.face_seed = face_seed if face_seed else random.randint(10000, 99999)
        
    def add_trait(self, trait):
        if trait not in self.traits:
            self.traits.append(trait)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "culture": self.culture,
            "age": self.age,
            "traits": self.traits,
            "voice_id": self.voice_id,
            "face_seed": self.face_seed
        }

    @staticmethod
    def from_dict(data):
        return Character(
            name=data["name"],
            culture=data["culture"],
            age=data["age"],
            traits=data["traits"],
            voice_id=data["voice_id"],
            face_seed=data["face_seed"]
        )

from config import MODEL_FAST
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class CharacterIdentity(BaseModel):
    name: str = Field(description="A culturally appropriate name for the protagonist")
    culture_label: str = Field(description="A formally normalized culture label (e.g., 'Indian Epic - Ramayana')")

class CharacterEngine:
    def __init__(self):
        self.llm = ChatGroq(model=MODEL_FAST)
        self.parser = JsonOutputParser(pydantic_object=CharacterIdentity)

    def _generate_identity_llm(self, theme_input):
        """Generates dynamic character identity using LLM."""
        try:
            prompt = (
                f"Analyze the theme '{theme_input}'.\n"
                "Generate a culturally authentic protagonist name and a formal culture label.\n"
                "Example: 'samurai' -> Name: 'Kenji', Culture: 'Japanese History - Samurai Era'\n"
                f"{self.parser.get_format_instructions()}"
            )
            response = self.llm.invoke(prompt)
            data = self.parser.parse(response.content)
            return data["name"], data["culture_label"]
        except Exception as e:
            print(f"Identity Generation Failed: {e}")
            return "Protagonist", theme_input.title()

    def initialize_character(self, name, culture_input):
        # If name is generic or missing, use LLM to generate identity
        if not name or "Protagonist" in name:
            gen_name, gen_culture = self._generate_identity_llm(culture_input)
            final_name = gen_name
            final_culture = gen_culture
        else:
            final_name = name
            final_culture = culture_input.title()
            
        return Character(final_name, final_culture)

    def get_visual_description(self, character):
        """Returns a stable visual description for the image prompt."""
        traits_str = ", ".join(character.traits)
        return (
            f"character {character.name}, {character.culture} ethnicity, "
            f"age {character.age}, wearing traditional attire, {traits_str}, "
            f"consistent face, high detail"
        )

    def update_traits_from_scores(self, character, scores):
        """Updates character traits based on moral alignment."""
        new_traits = []
        
        # Compassion
        c_score = scores.get('compassion', 0)
        if c_score >= 5:
            new_traits.append("Kind")
        elif c_score <= -5:
            new_traits.append("Ruthless")
            
        # Courage
        co_score = scores.get('courage', 0)
        if co_score >= 5:
            new_traits.append("Brave")
        elif co_score <= -5:
            new_traits.append("Cowardly")
            
        # Greed
        g_score = scores.get('greed', 0)
        if g_score >= 5:
            new_traits.append("Ambitious")
        elif g_score <= -5:
            new_traits.append("Generous")
            
        for trait in new_traits:
            character.add_trait(trait)
        
        return character
