import json
import os

from project import parse_command, save_conversation, load_conversation, trim_history

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

    with open(path, "r") as f:
        loaded = json.load(f)

    assert SAMPLE_MESSAGES == loaded


def test_saving_into_nonexistent_dir(tmp_path):
    nonexistent_path = os.path.join(tmp_path, "directory", "file.json")

    save_conversation(SAMPLE_MESSAGES, nonexistent_path)

    assert os.path.exists(nonexistent_path)


def test_load_conversation(tmp_path):
    path = os.path.join(tmp_path, "file.json")

    with open(path, "w") as f:
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
