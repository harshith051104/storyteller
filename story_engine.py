import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class StoryTeller:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
        self.history = []
        self.system_prompt = (
            "You are a 'Smart Cultural Storyteller'. Your goal is to preserve and retell cultural narratives, "
            "folk tales, and historical stories in an engaging, interactive 'choose-your-own-adventure' style. "
            "Always keep the responses concise (around 100-150 words) to keep the flow moving. "
            "End each response with 2 or 3 distinct choices for the user to decide how the story proceeds. "
            "If the user asks for a specific culture or theme, adapt entirely to that."
        )
        self.history.append(SystemMessage(content=self.system_prompt))

    def start_story(self, theme):
        """Initializes the story based on a theme."""
        # Reset history but keep system prompt
        self.history = [SystemMessage(content=self.system_prompt)]
        
        prompt = f"Start a story about {theme}. Set the scene and offer choices."
        self.history.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(self.history)
        self.history.append(response)
        
        return response.content

    def continue_story(self, user_choice):
        """Continues the story based on user's choice."""
        self.history.append(HumanMessage(content=user_choice))
        
        response = self.llm.invoke(self.history)
        self.history.append(response)
        
        return response.content
