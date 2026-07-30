import os
import anthropic

from dotenv import load_dotenv

load_dotenv()

# Access the api key stored as an environmental variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Constants
MAX_TOKENS = 1024
MODEL = "claude-haiku-4-5-20251001"


def main() -> None:
    print("Hello")
    
    while True:
        user_input = input("Prompt: ")
        
        if user_input == "/quit":
            break
        
        print(get_llm_response(user_input))
        
        
def get_llm_response(message: str) -> str:
    # These come from the anthropic's library documentation on handling errors
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        
        response = client.messages.create(
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ],
            model=MODEL
        )
    except anthropic.APIConnectionError:
        return "We apologise as the server could not be reached."
    # AuthenticationError and RateLimitError are subclasses of APIStatusError so must be caught first
    except anthropic.AuthenticationError:
        return "There has been an issue with the API key."
    except anthropic.RateLimitError:
        return "We apologise as there's too many requests at the minute."
    except anthropic.APIStatusError as e:
        return f"A status code of {e.status_code} was received."
    except anthropic.AnthropicError as e:
        return f"Something went wrong: {e}"
    
    if response.content and response.content[0].type == "text":
        return response.content[0].text
    
    return "No response received."
    

if __name__ == "__main__":
    main()