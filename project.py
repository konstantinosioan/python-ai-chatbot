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

COMMANDS = (
    "Available commands:\n"
    "/quit: to stop talking to chatbot\n"
    "/help: to see this printed list of commands\n"
    "/reset: to clear the conversation history and start fresh\n"
    "/quiz: to generate quiz questions based on discussion thus far\n"
    "/explain: to make the chatbot directly explain/answer your current question"
)


class ChatBot:
    def __init__(self) -> None:
        self.history: list[MessageParam] = []
        self.client = anthropic.Anthropic(api_key=API_KEY)

    def send_message(self, message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        self.history.append({"role": "user", "content": message})

        # These come from the anthropic's library documentation on handling errors
        try:
            response = get_llm_response(self.history, self.client, system_prompt)
        except anthropic.AnthropicError as e:
            return explain_llm_error(e)
        except ValueError:
            return "No response received."
        else:
            self.history.append({"role": "assistant", "content": response})

        return response

    def get_special_response(self, instruction: str, system_prompt: str) -> str:
        # Avoids appending the fake instruction to the history
        tmp_list = self.history + [{"role": "user", "content": instruction}]

        try:
            response = get_llm_response(tmp_list, self.client, system_prompt)
        except anthropic.AnthropicError as e:
            return explain_llm_error(e)
        except ValueError:
            return "No response received"
        else:
            self.history.append({"role": "assistant", "content": response})

        return response

    def reset_history(self) -> None:
        self.history.clear()

    def hold_conversation(self) -> None:
        print("Hello")

        while True:
            user_input = input("Prompt: ")

            result = parse_command(user_input)

            if result is None:
                response = self.send_message(user_input)
                print(response)
            else:
                command, argument = result

                if command == "/quit":
                    break
                else:
                    self.act_based_on_command(command, argument)

    def act_based_on_command(self, command: str, argument: str | None) -> None:
        match command:
            case "/reset":
                self.reset_history()
            case "/help":
                print(COMMANDS)
            case "/quiz":
                system_prompt = (
                    SYSTEM_PROMPT
                    + " For this response, generate a quiz based on the discussion thus far."
                )
                response = self.get_special_response(command, system_prompt)
                print(response)
            case "/explain":
                system_prompt = (
                    SYSTEM_PROMPT
                    + " For this response, fully explain the user's question."
                )
                response = self.get_special_response(command, system_prompt)
                print(response)


def main() -> None:
    try:
        chat_bot = ChatBot()
    except anthropic.AuthenticationError:
        print("There has been an issue with the API key.")
    else:
        chat_bot.hold_conversation()


def get_llm_response(
    messages: list[MessageParam], client: anthropic.Anthropic, system_prompt: str
) -> str:
    response = client.messages.create(
        max_tokens=MAX_TOKENS, messages=messages, model=MODEL, system=system_prompt
    )

    if response.content and response.content[0].type == "text":
        return response.content[0].text

    raise ValueError("No text content received from API.")


def parse_command(user_input: str) -> tuple[str, str | None] | None:
    # Only splits once and whitespace is the separator by default
    parts = user_input.split(maxsplit=1)

    if len(parts) == 0:
        return None
    elif len(parts) == 1:
        command = parts[0]
        argument = None
    else:
        command, argument = parts

    if command not in ("/quit", "/reset", "/help", "/quiz", "/explain"):
        return None

    return command, argument


def explain_llm_error(error: anthropic.AnthropicError) -> str:
    if isinstance(error, anthropic.APIConnectionError):
        return "We apologise as the server could not be reached."
    elif isinstance(error, anthropic.AuthenticationError):
        return "There has been an issue with the API key."
    elif isinstance(error, anthropic.RateLimitError):
        return "We apologise as there's too many requests at the minute."
    elif isinstance(error, anthropic.APIStatusError):
        return f"A status code of {error.status_code} was received."

    return f"Something went wrong: {error}"


if __name__ == "__main__":
    main()
