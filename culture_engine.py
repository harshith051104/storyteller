from config import MODEL_FAST
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from logger_config import get_logger

logger = get_logger()

class CultureEngine:
    def __init__(self):
        # Use Fast model for quick context retrieval/generation
        self.llm = ChatGroq(model=MODEL_FAST)

    def get_context_string(self, theme):
        """
        Dynamically generates a 'Knowledge Block' about the theme using the LLM.
        """
        if not theme:
            return ""

        logger.info(f"CultureEngine: Generatively recalling facts for '{theme}'...")
        
        system_prompt = (
            "You are an expert Cultural Anthropologist and Mythologist with encyclopedic knowledge of world cultures, "
            "folklore, and history. Your goal is to provide a concise, factual, and authentic 'Knowledge Block' "
            "that a storyteller can use to ground their narrative."
        )

        user_prompt = (
            f"Topic: {theme}\n\n"
            "Provide 9-10 key authentic cultural elements, including:\n"
            "1. Specific terminology (greetings, clothing, weapons, tools)\n"
            "2. Key festivals or rituals\n"
            "3. Mythological figures or legends\n"
            "4. Social hierarchy or values\n\n"
            "Format: A concise list or paragraph. Strictly factual and authentic. No preamble."
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Culture Generation Failed: {e}")
            return "General cultural knowledge applies."

if __name__ == "__main__":
    ce = CultureEngine()
    print(ce.get_context_string("Feudal Japan"))
