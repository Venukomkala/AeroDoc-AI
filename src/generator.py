import os
from groq import Groq


class AeroGenerator:

    def __init__(
        self,
        api_key=None,
        model_name="openai/gpt-oss-120b"
    ):

        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is missing."
            )

        self.client = Groq(
            api_key=self.api_key
        )

    def _build_system_prompt(self):

        return """
You are AeroDoc-AI, an expert technical assistant specializing
in aerospace document analysis.

Answer the user's question ONLY using the provided context.

Rules:

1. Ground every technical claim in the provided context.
2. Cite sources using:
   [Filename, Page X]
3. Do not invent facts.
4. Do not use outside knowledge.
5. If the context does not contain enough information, say:

"I cannot find sufficient information in the provided
technical documents to answer this query."

6. Give a clear and concise answer.
"""

    def generate_answer(
        self,
        query,
        context_text,
        temperature=0.2
    ):

        user_prompt = f"""
Context Information:

{context_text}

User Question:

{query}

Answer:
"""

        try:

            response = self.client.chat.completions.create(

                model=self.model_name,

                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            print(
                f"[Primary model failed]: {e}"
            )

            # Current fallback model
            try:

                response = self.client.chat.completions.create(

                    model="openai/gpt-oss-20b",

                    messages=[
                        {
                            "role": "system",
                            "content": self._build_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    temperature=temperature
                )

                return response.choices[0].message.content.strip()

            except Exception as fallback_error:

                print(
                    f"[Fallback model failed]: "
                    f"{fallback_error}"
                )

                return (
                    "LLM generation failed. "
                    "Please check your Groq API key "
                    "and model access."
                )