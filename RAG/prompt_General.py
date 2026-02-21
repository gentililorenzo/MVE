@staticmethod
def promptGeneral(user_question: str):
    return f"""
# ROLE
You are a senior Sustainability Expert.

<user_question>
{user_question}
</user_question>
"""