"""A Streamlit web UI for the Socratic study-assistant chatbot, reusing project.py's ChatBot class"""

import os
import json
import tempfile
from collections.abc import Callable

import streamlit as st
import anthropic

from project import ChatBot, API_KEY_ERROR_MESSAGE, explain_llm_error, SYSTEM_PROMPT
from rag import Retriever

st.title("Study Assistant", text_alignment="center")

if "chat_bot" not in st.session_state:
    try:
        st.session_state.chat_bot = ChatBot()
    except ValueError:
        st.error(API_KEY_ERROR_MESSAGE)
        st.stop()


def display_and_clear(key: str, display_function: Callable[[str], object]) -> None:
    """
    Displays a pending message stored in session_state under the given key and then clears it if one exists

    :param key: The session_state key that may hold a pending message
    :param display_function: The Streamlit function used to display the message (e.g. st.error, st.success)
    """
    if st.session_state.get(key):
        display_function(st.session_state[key])
        st.session_state[key] = None


# Display chat messages from history on app rerun
for message in st.session_state.chat_bot.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

display_and_clear("instruction_error", st.error)

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = st.session_state.chat_bot.retrieve_context(prompt, SYSTEM_PROMPT)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            st.write_stream(
                st.session_state.chat_bot.stream_response(prompt, system_prompt)
            )
        except anthropic.AnthropicError as e:
            st.error(explain_llm_error(e))
        except ValueError:
            st.error("No response received.")


def handle_conversation_upload() -> None:
    """Used as a file_uploader callback for /load. Restores the uploaded conversation as the current history and stores the result message for display"""
    conversation_file = st.session_state.conversation_upload

    if conversation_file is None or conversation_file.name == st.session_state.get(
        "conversation_filename"
    ):
        return

    st.session_state.conversation_filename = conversation_file.name

    try:
        st.session_state.chat_bot.history = json.loads(conversation_file.getvalue())
    except UnicodeDecodeError:
        st.session_state.conversation_upload_error = "The file is not UTF-8 encoded."
    except json.JSONDecodeError:
        st.session_state.conversation_upload_error = (
            "The file is not a valid JSON document."
        )
    else:
        st.session_state.conversation_upload_success = (
            "Conversation successfully loaded."
        )


st.sidebar.file_uploader(
    "Load a saved conversation",
    type=["json"],
    key="conversation_upload",
    on_change=handle_conversation_upload,
)

display_and_clear("conversation_upload_error", st.error)
display_and_clear("conversation_upload_success", st.success)

# /save equivalent
if st.session_state.chat_bot.history:
    conversation_json = json.dumps(st.session_state.chat_bot.history, indent=2)

    st.sidebar.download_button(
        label="Download conversation",
        data=conversation_json,
        file_name="conversation.json",
        mime="application/json",
    )


def handle_document_upload() -> None:
    """
    Used as a file_uploader callback for /loaddoc. Indexes the newly uploaded document
    into the retriever and stores the result message for display; clears the retriever if the file is removed
    """
    uploaded_file = st.session_state.document_upload

    if uploaded_file is None:
        st.session_state.chat_bot.retriever = None
        st.session_state.loaded_filename = None
        return

    if uploaded_file.name == st.session_state.get("loaded_filename"):
        return

    st.session_state.loaded_filename = uploaded_file.name

    if st.session_state.chat_bot.retriever is None:
        try:
            st.session_state.chat_bot.retriever = Retriever()
        except ValueError:
            st.session_state.document_upload_error = API_KEY_ERROR_MESSAGE
            return

    _, suffix = os.path.splitext(uploaded_file.name)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    result = st.session_state.chat_bot.retriever.index_document(tmp_path)
    os.remove(tmp_path)

    if isinstance(result, str):
        st.session_state.document_upload_error = result
    else:
        st.session_state.document_upload_success = "File successfully loaded."


def handle_reset_click() -> None:
    """Used as a button callback. Clears conversation history, forgets which files were already processed and clears the loaded document's retriever"""
    st.session_state.chat_bot.reset_history()
    st.session_state.chat_bot.retriever = None
    st.session_state.loaded_filename = None
    st.session_state.conversation_filename = None


def handle_instruction_button(command: str, instruction: str) -> None:
    """
    Used as a button callback. Builds an extended system prompt from the instruction and gets the full response, storing an error message for display if the call fails.

    :param command: One of the existing commands; sent as the message content to the LLM
    :param instruction: The instruction to be concatenated with the base system prompt to produce different behaviour
    """
    system_prompt = SYSTEM_PROMPT + " For this response, " + instruction

    try:
        # This is not streamed as it runs as a callback before the chat_bot.history display loop re-executes
        # so streaming here would render in the wrong place and get duplicated
        for _ in st.session_state.chat_bot.stream_response(command, system_prompt):
            pass
    except anthropic.AnthropicError as e:
        st.session_state.instruction_error = explain_llm_error(e)
    except ValueError:
        st.session_state.instruction_error = "No response received."


st.sidebar.file_uploader(
    "Load a document",
    type=["txt", "md", "pdf"],
    key="document_upload",
    on_change=handle_document_upload,
)

display_and_clear("document_upload_error", st.error)
display_and_clear("document_upload_success", st.success)


# The following buttons use callback functions so their state changes apply before the page reruns

# /reset equivalent
st.sidebar.button("Reset conversation", on_click=handle_reset_click)

# /quiz equivalent
st.sidebar.button(
    "Quiz me",
    on_click=handle_instruction_button,
    args=("/quiz", "generate a quiz based on the discussion thus far."),
)

# /explain equivalent
st.sidebar.button(
    "Explain",
    on_click=handle_instruction_button,
    args=("/explain", "fully explain the user's question."),
)

# /summarise equivalent
st.sidebar.button(
    "Summarise",
    on_click=handle_instruction_button,
    args=(
        "/summarise",
        "summarise what's been covered in this conversation thus far and what the user should review.",
    ),
)
