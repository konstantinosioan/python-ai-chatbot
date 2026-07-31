"""Tests for project.py"""

import json
import os

from unittest.mock import patch

import anthropic

from project import (
    parse_command,
    save_conversation,
    load_conversation,
    trim_history,
    ChatBot,
)

SAMPLE_MESSAGES = [
    {"role": "user", "content": "wassup"},
    {"role": "assistant", "content": "Hey!"},
    {"role": "user", "content": "Are you good?"},
    {"role": "assistant", "content": "Yes!"},
    {"role": "assistant", "content": "Bye!"},
]


def test_parse_command():
    assert parse_command("/save file") == ("/save", "file")


def test_valid_command_without_argument():
    assert parse_command("/quit") == ("/quit", None)


def test_valid_command_with_argument():
    assert parse_command("/quit this is my program") == ("/quit", "this is my program")


def test_normal_input():
    assert parse_command("hello there") is None


def test_empty_input():
    assert parse_command("") is None


def test_invalid_command():
    assert parse_command("/bogus") is None


def test_valid_command_with_whitespace():
    assert parse_command(" /quit ") == ("/quit", None)


def test_save_conversation(tmp_path):
    path = os.path.join(tmp_path, "file.json")
    save_conversation(SAMPLE_MESSAGES, path)

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert SAMPLE_MESSAGES == loaded


def test_saving_into_nonexistent_dir(tmp_path):
    nonexistent_path = os.path.join(tmp_path, "directory", "file.json")

    save_conversation(SAMPLE_MESSAGES, nonexistent_path)

    assert os.path.exists(nonexistent_path)


def test_load_conversation(tmp_path):
    path = os.path.join(tmp_path, "file.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_MESSAGES, f)

    loaded = load_conversation(path)

    assert SAMPLE_MESSAGES == loaded


def test_trim_history():
    messages = [
        {"role": "user", "content": "wassup"},
        {"role": "assistant", "content": "Hey!"},
    ]

    expected_trimmed_list = []

    assert expected_trimmed_list == trim_history(messages, 1)


def test_history_less_than_limit():
    assert SAMPLE_MESSAGES == trim_history(SAMPLE_MESSAGES, 6)


def test_history_same_size_as_limit():
    assert SAMPLE_MESSAGES == trim_history(SAMPLE_MESSAGES, 5)


def test_empty_history():
    assert [] == trim_history([], 3)


def test_trimmed_list_starts_with_user_prompt():
    expected_trimmed_list = [
        {"role": "user", "content": "Are you good?"},
        {"role": "assistant", "content": "Yes!"},
        {"role": "assistant", "content": "Bye!"},
    ]

    assert expected_trimmed_list == trim_history(SAMPLE_MESSAGES, 3)


def test_trimmed_list_starts_with_one_assistant_entry():
    expected_trimmed_list = [
        {"role": "user", "content": "Are you good?"},
        {"role": "assistant", "content": "Yes!"},
        {"role": "assistant", "content": "Bye!"},
    ]

    assert expected_trimmed_list == trim_history(SAMPLE_MESSAGES, 4)


def test_multiple_leading_assistant_entries():
    expected_trimmed_list = []

    assert expected_trimmed_list == trim_history(SAMPLE_MESSAGES, 2)


def test_send_message_success():
    with patch("project.get_llm_response") as mock_function:
        mock_function.return_value = "reply"

        chat_bot = ChatBot()
        response = chat_bot.send_message("hello")

    assert response == "reply"
    assert chat_bot.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]


def test_send_message_generic_anthropic_error():
    with patch("project.get_llm_response") as mock_function:
        mock_function.side_effect = anthropic.AnthropicError("test")

        chat_bot = ChatBot()
        response = chat_bot.send_message("hello")

    assert response == "Something went wrong: test"
    assert chat_bot.history == []


def test_send_message_empty_response():
    with patch("project.get_llm_response") as mock_function:
        mock_function.side_effect = ValueError("test")

        chat_bot = ChatBot()
        response = chat_bot.send_message("hello")

    assert response == "No response received."
    assert chat_bot.history == []


def test_load_file_not_found(tmp_path, capsys):
    chat_bot = ChatBot()
    nonexistent_path = os.path.join(tmp_path, "directory", "file.json")
    chat_bot.act_based_on_command("/load", nonexistent_path)

    output = capsys.readouterr()

    assert output.out == "File not found.\n"
    assert chat_bot.history == []


def test_load_invalid_json(tmp_path, capsys):
    chat_bot = ChatBot()
    path = os.path.join(tmp_path, "file.json")

    with open(path, "w", encoding="utf-8") as f:
        f.write("non json text")

    chat_bot.act_based_on_command("/load", path)

    output = capsys.readouterr()

    assert output.out == "The file is not a valid JSON document.\n"
    assert chat_bot.history == []


def test_load_invalid_encoding(tmp_path, capsys):
    chat_bot = ChatBot()
    path = os.path.join(tmp_path, "file.json")

    with open(path, "wb") as f:
        f.write(b"\xff")

    chat_bot.act_based_on_command("/load", path)

    output = capsys.readouterr()

    assert (
        output.out
        == "The file does not contain UTF-8, UTF-16 or UTF-32 encoded data.\n"
    )
    assert chat_bot.history == []


def test_reset_history():
    chat_bot = ChatBot()
    chat_bot.history = [
        {"role": "user", "content": "wassup"},
        {"role": "assistant", "content": "Hey!"},
    ]
    chat_bot.reset_history()

    assert chat_bot.history == []
