from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_community.callbacks import get_openai_callback
from langdetect import detect, LangDetectException

class InsuranceChatbot:
    """An expert US insurance advisor chatbot using a controlled knowledge base and strict prompting."""

    INSURANCE_KNOWLEDGE = """
    # US Insurance Knowledge Base

    ## Health Insurance
    - Types: HMO, PPO, EPO, HDHP
    - Key terms: Premium, Deductible, Copay, Coinsurance, Out-of-pocket maximum
    - Special programs: Medicare (Parts A-D), Medicaid, ACA marketplace plans

    ## Auto Insurance
    - Common coverage types: Liability, Collision, Comprehensive, UM/UIM, PIP
    - State minimum requirements vary
    - Factors affecting rates: driving record, vehicle type, location

    ## Homeowners Insurance
    - Standard coverage: Dwelling, Personal Property, Liability, ALE
    - Common exclusions: Flood, Earthquake (require separate policies)
    - HO-3 is most common policy type

    ## Life Insurance
    - Term vs. Whole life
    - Key concepts: Death benefit, Cash value, Underwriting
    - Typical coverage amounts based on income replacement needs
    """

    # Multilingual out-of-scope messages
    OUT_OF_SCOPE_MESSAGES = {
        'en': "I can only assist with US insurance topics from my knowledge base.",
        'es': "Solo puedo ayudar con temas de seguros de EE. UU. de mi base de conocimientos.",
        'hi': "मैं केवल अपने ज्ञान आधार से यूएस बीमा विषयों में सहायता कर सकता हूं।",
        # Add more languages as needed
    }

    DEFAULT_LANGUAGE = 'en'

    def __init__(self, api_key: str):
        """Initialize the insurance chatbot with the OpenAI LLM."""
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",  # Changed from "gpt-4o-mini" which doesn't exist
            temperature=0.1,
        )
        
        # Initialize with default language prompt
        self.update_system_prompt(self.DEFAULT_LANGUAGE)

    def update_system_prompt(self, language: str):
        """Update the system prompt with the appropriate language."""
        out_of_scope_msg = self.OUT_OF_SCOPE_MESSAGES.get(language, self.OUT_OF_SCOPE_MESSAGES['en'])
        
        self.SYSTEM_PROMPT = SystemMessage(content=f"""
        You are an expert US insurance advisor. Your answers must strictly follow this structured knowledge base:

        {self.INSURANCE_KNOWLEDGE}

        INSTRUCTIONS:
        1. ONLY use the information provided above.
        2. DO NOT answer questions outside this knowledge. Instead say:
           "{out_of_scope_msg}"
        3. NEVER guess, invent or assume beyond the listed content.
        4. Your response must be clear, helpful, and NEVER shorter than 25 words.
        5. You must act as a helpful assistant—never robotic or vague.

        """)
        
        # Rebuild the chain with the updated prompt
        self.prompt_template = ChatPromptTemplate.from_messages([
            self.SYSTEM_PROMPT,
            ("human", "{user_query}")
        ])

        self.chain = (
            {"user_query": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            lang = detect(text)
            return lang if lang in self.OUT_OF_SCOPE_MESSAGES else self.DEFAULT_LANGUAGE
        except LangDetectException:
            return self.DEFAULT_LANGUAGE

    def get_insurance_response(self, user_query: str) -> str:
        """Get a structured insurance-related response."""
        try:
            # Detect language and update prompt if needed
            lang = self.detect_language(user_query)
            self.update_system_prompt(lang)
            
            with get_openai_callback() as cb:
                response = self.chain.invoke(user_query)
                token_usage = cb.total_tokens
                print(f"Tokens used: {token_usage}")

            return response
        except Exception as e:
            lang = self.detect_language(user_query)
            error_msg = self.OUT_OF_SCOPE_MESSAGES.get(lang, self.OUT_OF_SCOPE_MESSAGES['en'])
            return f"{error_msg} [System Error: {str(e)}]"

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

    chatbot = InsuranceChatbot(api_key)

    questions = [
        "I'm going to retire next year. What should I know about health insurance?",
        "What's the difference between term and whole life insurance?",
        "¿Qué tipos de seguro de auto existen en Estados Unidos?",
        "मुझे स्वास्थ्य बीमा के बारे में जानकारी चाहिए",
        "Do you provide advice on travel insurance?",  # triggers out-of-scope rule
        "¿Puedes ayudarme con un seguro de viaje?",  # triggers out-of-scope rule in Spanish
        "क्या आप यात्रा बीमा के बारे में सलाह देते हैं?"  # triggers out-of-scope rule in Hindi
    ]

    for question in questions:
        print(f"Q: {question}")
        print(f"A: {chatbot.get_insurance_response(question)}\n")
        print("-" * 80)