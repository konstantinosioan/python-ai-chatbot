import os
import anthropic
from anthropic.types import MessageParam

from dotenv import load_dotenv

load_dotenv()

# Access the api key stored as an environmental variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Constants
MAX_TOKENS = 1024
MODEL = "claude-haiku-4-5-20251001"

# Configure how the AI works
SYSTEM_PROMPT = (
    "Your job is to help the user with their studies. "
    "You are to use the Socratic method of teaching, that is to guide the user by asking them questions, avoid giving them direct answers and making them think. "
    "You may only give away hints but always try to get the user to attempt and reason first, by asking the user to explain their thinking. "
    "Confirm they understand something before continuing. "
    "Where the user has made a genuine respectable attempt, even if not fully correct, you may explain directly, but for shaky attempts/weak arguments do challenge them. "
    "If the user indicates it's a formal assignment, you may push back on helping them."
)


class ChatBot:
    def __init__(self) -> None:
        self.history: list[MessageParam] = []
        self.client = anthropic.Anthropic(api_key=API_KEY)

    def send_message(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})

        # These come from the anthropic's library documentation on handling errors
        try:
            response = get_llm_response(self.history, self.client)
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
        except ValueError:
            return "No response received."
        else:
            self.history.append({"role": "assistant", "content": response})

        return response

    def reset_history(self) -> None:
        self.history.clear()

    def hold_conversation(self) -> None:
        print("Hello")

        while True:
            user_input = input("Prompt: ")

            if user_input == "/quit":
                break

            response = self.send_message(user_input)

            print(response)


def main() -> None:
    try:
        chat_bot = ChatBot()
    except anthropic.AuthenticationError:
        print("There has been an issue with the API key.")
    else:
        chat_bot.hold_conversation()


def get_llm_response(messages: list[MessageParam], client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        max_tokens=MAX_TOKENS, messages=messages, model=MODEL, system=SYSTEM_PROMPT
    )

    if response.content and response.content[0].type == "text":
        return response.content[0].text

    raise ValueError("No text content received from API.")


if __name__ == "__main__":
    main()
