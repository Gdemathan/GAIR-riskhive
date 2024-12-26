from openai import OpenAI
import os

try:
    import dotenv
    dotenv.load_dotenv()
except:
    pass

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))