"""A Command-line Interface Socratic-tutor chatbot using the Anthropic API"""

import os
import json
from collections.abc import Iterator
import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from rag import Retriever, augment_system_prompt

load_dotenv()

# Access the API key stored as an environment variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Constants
MAX_TOKENS = 1024
MODEL = "claude-haiku-4-5-20251001"
CONVERSATIONS_DIR = "conversations/"

# Used to trim the message history to not waste as many tokens
HISTORY_LIMIT = 30

# Shared user-facing error/hint messages
API_KEY_ERROR_MESSAGE = "There has been an issue with the API key."
USAGE_HINT_MESSAGE = "Please check correct usage with the /help command."

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
    "/explain: to make the chatbot directly explain/answer your current question\n"
    "/summarise: to make the chatbot generate a summary of the discussion so far\n"
    "/save [filename]: to save the conversation to conversations/[filename] in json format\n"
    "/load [path]: to load a previously saved conversation from given path\n"
    "/loaddoc [path]: to load a document from given path so questions can be answered based on it"
)


class ChatBot:
    """The stateful core - Message history, client initialisation, retriever/document state, command specific behaviour"""

    def __init__(self) -> None:
        """
        Initialises an empty list for the message history, sets up the client and leaves the retriever unset until a document is loaded

        :raise ValueError: If the API key is missing
        """
        if API_KEY is None:
            raise ValueError("No API key.")

        self.history: list[MessageParam] = []
        self._client = anthropic.Anthropic(api_key=API_KEY)
        self.retriever: Retriever | None = None

    def send_message(self, message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Sends message to LLM, first grounding the system prompt in the loaded document's content if one exists.
        On success, both the prompt and response get appended to history and the history is trimmed.
        On failure, it prints and returns an error message without touching the history.

        :param message: The prompt from the user
        :param system_prompt: Optional argument. Used for configuring the chatbot's underlying behaviour and specifically making it act as a tutor
        :return: The response from the chatbot on success; Error message on failure
        """
        # Avoids appending a user's message to history if something goes wrong
        tmp_list = self.history + [{"role": "user", "content": message}]

        # Grounds the prompt in the loaded document's content, if one exists
        system_prompt = self.retrieve_prompt_with_context(message, system_prompt)

        print("\nTutor: ", end="")

        # These come from the anthropic's library documentation on handling errors
        try:
            response = get_llm_response(tmp_list, self._client, system_prompt)
        except anthropic.AnthropicError as e:
            error_message = explain_llm_error(e)
            print(error_message)
            return error_message
        except ValueError:
            error_message = "No response received."
            print(error_message)
            return error_message

        print()

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.history = trim_history(self.history, HISTORY_LIMIT)

        return response

    def reset_history(self) -> None:
        """Clears the conversation history"""
        self.history.clear()

    def hold_conversation(self) -> None:
        """The main loop: greets, reads user input and skips it if blank and checks for command use or routes to chat. Exits gracefully on /quit or Ctrl+C"""
        print(
            "Welcome to your Socratic study assistant. It will guide you towards the answers to your questions without handing them over. Type /help to see available commands."
        )

        while True:
            try:
                user_input = input("\nPrompt: ").strip()

                if not user_input:
                    continue

                result = parse_command(user_input)

                if result is None:
                    self.send_message(user_input)
                else:
                    command, argument = result

                    if command == "/quit":
                        break

                    self.act_based_on_command(command, argument)
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break

    def act_based_on_command(self, command: str, argument: str | None) -> None:
        """
        Matches each recognised command to a specific action

        :param command: One of the existing commands
        :param argument: An argument if there is one or None if there isn't
        """
        match command:
            case "/reset":
                self.reset_history()
            case "/help":
                print(COMMANDS)
            case "/quiz":
                self.respond_with_instruction(
                    command, "generate a quiz based on the discussion thus far."
                )
            case "/explain":
                self.respond_with_instruction(
                    command, "fully explain the user's question."
                )
            case "/summarise":
                self.respond_with_instruction(
                    command,
                    "summarise what's been covered in this conversation thus far and what the user should review.",
                )
            case "/save":
                if argument is not None:
                    try:
                        # Saving enforces one consistent directory
                        path = os.path.join(CONVERSATIONS_DIR, argument)
                        save_conversation(self.history, path)
                    except PermissionError:
                        print("You don't have permission for that.")
                    except OSError as e:
                        print(f"Something went wrong with the operation: {e}")
                    else:
                        print("Conversation successfully saved as a JSON file.")
                else:
                    print(USAGE_HINT_MESSAGE)
            case "/load":
                if argument is not None:
                    try:
                        # Loading accepts any path the user wants
                        self.history = load_conversation(argument)
                    except FileNotFoundError:
                        print("File not found.")
                    except OSError as e:
                        print(f"Something went wrong with the operation: {e}")
                    except json.JSONDecodeError:
                        print("The file is not a valid JSON document.")
                    except UnicodeDecodeError:
                        print("The file is not UTF-8 encoded.")
                    else:
                        print("Conversation successfully loaded from file.")
                else:
                    print(USAGE_HINT_MESSAGE)
            case "/loaddoc":
                if argument is not None:
                    if self.retriever is None:
                        try:
                            self.retriever = Retriever()
                        except ValueError:
                            print(API_KEY_ERROR_MESSAGE)
                            return

                    result = self.retriever.index_document(argument)

                    if isinstance(result, str):
                        print(result)
                    else:
                        print("File successfully loaded.")
                else:
                    print(USAGE_HINT_MESSAGE)

    def respond_with_instruction(self, command: str, instruction: str) -> None:
        """
        Shared helper for commands that need the LLM to do something by building an extended system prompt and sending it

        :param command: One of the existing commands; sent as the message content to the LLM
        :param instruction: The instruction to be concatenated with the base system prompt to produce different behaviour
        """
        system_prompt = SYSTEM_PROMPT + " For this response, " + instruction
        self.send_message(command, system_prompt)

    def retrieve_prompt_with_context(self, message: str, system_prompt: str) -> str:
        """
        Returns system prompt extended with additional context retrieved from loaded document, printing the retrieved chunks first.
        Falls back to the given system prompt unchanged if no document is loaded or retrieval fails.

        :param message: The user's current message; combined with recent history to form the retrieval query
        :param system_prompt: The base system prompt to augment
        :return: The augmented system prompt if retrieval succeeds; the original system prompt otherwise
        """
        augmented_prompt, result = self._retrieve_augmented_prompt(
            message, system_prompt
        )

        if result is None:
            return augmented_prompt

        # If something fails during retrieval
        if isinstance(result, str):
            print(f"(Couldn't retrieve context from the loaded document: {result})")
            return augmented_prompt

        print("Found the following in the loaded document:")
        for chunk in result:
            print(f"  - {chunk[:70]}...")

        return augmented_prompt

    def retrieve_context(self, message: str, system_prompt: str) -> str:
        """
        Used by the Streamlit UI (app.py), not the CLI.
        Returns system prompt extended with additional context retrieved from loaded document.
        Falls back to the given system prompt unchanged if no document is loaded or retrieval fails.

        :param message: The user's current message; combined with recent history to form the retrieval query
        :param system_prompt: The base system prompt to augment
        :return: The augmented system prompt if retrieval succeeds; the original system prompt otherwise
        """
        augmented_prompt, _ = self._retrieve_augmented_prompt(message, system_prompt)

        return augmented_prompt

    def _retrieve_augmented_prompt(
        self, message: str, system_prompt: str
    ) -> tuple[str, None | list[str] | str]:
        """
        Shared retrieval logic used by both the printing and non-printing variants

        :param message: The user's current message; combined with recent history to form the retrieval query
        :param system_prompt: The base system prompt to augment
        :return: A tuple consisting of the system prompt (augmented on success and unchanged on failure)
            and either the list of chunks on success, an error message on failure or None if no document is loaded
        """
        if self.retriever is None or not self.retriever.chunks:
            return system_prompt, None

        result = self.retriever.retrieve(build_retrieval_query(message, self.history))

        if isinstance(result, str):
            return system_prompt, result

        return augment_system_prompt(system_prompt, result), result

    def stream_response(
        self, message: str, system_prompt: str = SYSTEM_PROMPT
    ) -> Iterator[str]:
        """
        Used by the Streamlit UI (app.py), not the CLI.
        Sends message and system prompt to API, yielding response text chunks as they arrive.
        On success, appends the user message and full response to history and trims it. On failure, it raises an exception without touching history

        :param message: The prompt from the user
        :param system_prompt: Optional argument. Used for configuring the chatbot's underlying behaviour
        :raise anthropic.APIConnectionError: If any network connection failures occur
        :raise anthropic.AuthenticationError: If the API key is invalid, expired or revoked
        :raise anthropic.RateLimitError: If the rate limit is exceeded
        :raise anthropic.APIStatusError: If any HTTP errors occur
        :raise anthropic.AnthropicError: If anything other than the above goes wrong regarding the API
        :raise ValueError: If the response is empty or has non-text content
        :yield: Chunks of the response text as they arrive
        """
        tmp_list = self.history + [{"role": "user", "content": message}]
        chunks = []

        with self._client.messages.stream(
            max_tokens=MAX_TOKENS,
            messages=tmp_list,
            model=MODEL,
            system=system_prompt,
        ) as stream:
            for text in stream.text_stream:
                yield text
                chunks.append(text)

            response = stream.get_final_message()

        if not (response.content and response.content[0].type == "text"):
            raise ValueError("No text content received from API.")

        full_response = "".join(chunks)
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})
        self.history = trim_history(self.history, HISTORY_LIMIT)


def main() -> None:
    """The entry point of the program - Constructs the ChatBot object and starts the conversation loop"""
    try:
        chat_bot = ChatBot()
    except ValueError:
        print(API_KEY_ERROR_MESSAGE)
    else:
        chat_bot.hold_conversation()


def get_llm_response(
    messages: list[MessageParam], client: anthropic.Anthropic, system_prompt: str
) -> str:
    """
    Sends message history and system prompt to API, streaming and printing the response text as it arrives. On success, returns the full response. On failure, it raises an exception

    :param messages: The list containing message history
    :param client: The Anthropic client
    :param system_prompt: The string of text used to configure the chatbot's underlying behaviour
    :raise anthropic.APIConnectionError: If any network connection failures occur
    :raise anthropic.AuthenticationError: If the API key is invalid, expired or revoked
    :raise anthropic.RateLimitError: If the rate limit is exceeded
    :raise anthropic.APIStatusError: If any HTTP errors occur
    :raise anthropic.AnthropicError: If anything other than the above goes wrong regarding the API
    :raise ValueError: If the response is empty or has non-text content
    :return: The response string provided it comes back fine
    """
    with client.messages.stream(
        max_tokens=MAX_TOKENS,
        messages=messages,
        model=MODEL,
        system=system_prompt,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

        response = stream.get_final_message()

    if response.content and response.content[0].type == "text":
        return response.content[0].text

    raise ValueError("No text content received from API.")


def parse_command(user_input: str) -> tuple[str, str | None] | None:
    """
    Extracts the command and argument (if there is one) and returns them. If the command doesn't exist or user input is empty, None is returned.

    :param user_input: The user prompt
    :return: A tuple containing a command (if matched with one) and an argument (None if not provided). If unrecognised command or empty input, None is returned
    """
    # Only splits once and whitespace is the separator by default
    parts = user_input.split(maxsplit=1)

    if len(parts) == 0:
        return None
    elif len(parts) == 1:
        command = parts[0]
        argument = None
    else:
        command, argument = parts

    if command not in (
        "/quit",
        "/reset",
        "/help",
        "/quiz",
        "/explain",
        "/summarise",
        "/save",
        "/load",
        "/loaddoc",
    ):
        return None

    return command, argument


def explain_llm_error(error: anthropic.AnthropicError) -> str:
    """
    Returns a friendly error message for each caught anthropic exception

    :param error: The general AnthropicError caught
    :return: A friendly error message
    """
    if isinstance(error, anthropic.APIConnectionError):
        return "We apologise as the server could not be reached."
    if isinstance(error, anthropic.AuthenticationError):
        return API_KEY_ERROR_MESSAGE
    if isinstance(error, anthropic.RateLimitError):
        return "We apologise as there's too many requests at the minute."
    if isinstance(error, anthropic.APIStatusError):
        return f"A status code of {error.status_code} was received."

    return f"Something went wrong: {error}"


def save_conversation(messages: list[MessageParam], path: str) -> None:
    """
    Writes the message history list to a JSON file at a given path, creating the parent directories if needed

    :param messages: The list containing message history
    :param path: The path at which to create and store the file
    :raise PermissionError: If user does not have the necessary permissions to write to the given path
    :raise OSError: If any other thing goes wrong when saving conversation
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)


def load_conversation(path: str) -> list[MessageParam]:
    """
    Reads a JSON file at a given path into the list containing message history

    :param path: The path at which the file to read from lives
    :raise FileNotFoundError: If the file cannot be found
    :raise OSError: If any other thing goes wrong when reading file
    :raise json.JSONDecodeError: If the file is not a valid JSON document
    :raise UnicodeDecodeError: If the file is not UTF-8 encoded
    :return: The message history read from the file
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trim_history(messages: list[MessageParam], limit: int) -> list[MessageParam]:
    """
    Trims the history list to the most recent `limit` entries. Strips any leading assistant replies to preserve user-assistant pairs

    :param messages: The list containing message history
    :param limit: The number of most recent messages to keep
    :return: The trimmed list containing message history
    """
    # Gets the most recent limit messages
    trimmed = messages[-limit:]

    while trimmed and trimmed[0]["role"] == "assistant":
        # Trim off any leading assistant message
        trimmed = trimmed[1:]

    return trimmed


def build_retrieval_query(current_message: str, history: list[MessageParam]) -> str:
    """
    Builds a string containing the user's current message and their last two exchanges with the assistant so that it has more context and returns it

    :param current_message: The current prompt of the user
    :param history: The list containing message history
    :return: String containing the previous two user-assistant exchanges and the user's current message
    """
    previous_two_exchanges = history[-4:]

    return (
        " ".join(message["content"] for message in previous_two_exchanges)
        + " "
        + current_message
    )


if __name__ == "__main__":
    main()
