from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from config import MODEL_FAST, MORAL_SCORE_MIN, MORAL_SCORE_MAX
from logger_config import get_logger

logger = get_logger()

class MoralScore(BaseModel):
    compassion: int = Field(description="Change in compassion score (-5 to +5)")
    courage: int = Field(description="Change in courage score (-5 to +5)")
    greed: int = Field(description="Change in greed score (-5 to +5)")
    reasoning: str = Field(description="Brief reason for the score")

class MoralEngine:
    def __init__(self):
        self.llm = ChatGroq(model=MODEL_FAST, temperature=0.5)
        self.scores = {"compassion": 0, "courage": 0, "greed": 0}
        
        self.parser = JsonOutputParser(pydantic_object=MoralScore)

    def score_choice(self, user_choice, story_context):
        """Evaluates the user's last choice."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Moral Arbiter in a story game. Analyze the user's choice and assign score changes."),
            ("human", "Story Context: {context}\nUser Choice: {choice}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "context": story_context[-500:], # Last 500 chars context
                "choice": user_choice,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Update internal state with clamping
            for trait in ["compassion", "courage", "greed"]:
                change = result.get(trait, 0)
                new_score = self.scores[trait] + change
                # Clamp score
                self.scores[trait] = max(MORAL_SCORE_MIN, min(MORAL_SCORE_MAX, new_score))
            
            return result
        except Exception as e:
            logger.error(f"Moral Engine Error: {e}")
            return None

    def generate_reflection(self):
        """Generates a final moral summary."""
        prompt = f"""
        Based on these final scores: {self.scores},
        write a 2-sentence spiritual reflection for the player, referencing concepts like Karma or Dharma if appropriate.
        """
        response = self.llm.invoke(prompt)
        return response.content
