"""Tests for rag.py"""

import os
from unittest.mock import patch

import pytest
import voyageai.error as verr

import rag
from rag import (
    split_text_into_chunks,
    explain_api_error,
    augment_system_prompt,
    read_file_as_string,
    Retriever,
)


def test_split_text_into_chunks():
    assert split_text_into_chunks("paragraph1\n\nparagraph2\n\nparagraph3") == [
        "paragraph1",
        "paragraph2",
        "paragraph3",
    ]


def test_split_text_single_paragraph():
    assert split_text_into_chunks("paragraph") == ["paragraph"]


def test_split_empty_string():
    assert split_text_into_chunks("") == []


def test_split_text_extra_blank_lines():
    assert split_text_into_chunks("paragraph1\n\n\n\nparagraph2") == [
        "paragraph1",
        "paragraph2",
    ]


def test_split_text_extra_whitespace():
    assert split_text_into_chunks(" paragraph  ") == ["paragraph"]


def test_explain_api_connection_error():
    assert (
        explain_api_error(verr.APIConnectionError())
        == "We apologise as the server could not be reached."
    )


def test_explain_api_authentication_error():
    assert (
        explain_api_error(verr.AuthenticationError())
        == "There has been an issue with the API key."
    )


def test_explain_api_rate_limit_error():
    assert (
        explain_api_error(verr.RateLimitError())
        == "We apologise as there's too many requests at the minute."
    )


def test_explain_api_unhandled_error():
    assert (
        explain_api_error(verr.ServerError("server error"))
        == "Something went wrong: server error"
    )


def test_augment_system_prompt():
    system_prompt = "some prompt"
    chunks = ["par1", "par2", "par3"]

    result = augment_system_prompt(system_prompt, chunks)

    assert system_prompt in result
    assert "\n\n".join(chunks) in result


def test_augment_system_prompt_no_chunks():
    system_prompt = "some prompt"
    result = augment_system_prompt(system_prompt, [])
    assert system_prompt in result


def test_augment_system_prompt_one_chunk():
    system_prompt = "some prompt"
    chunks = ["paragraph"]

    result = augment_system_prompt(system_prompt, chunks)

    assert system_prompt in result
    assert "paragraph" in result


def test_read_file_as_string(tmp_path):
    path = os.path.join(tmp_path, "file.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("hello there")
        f.write("well goodbye")

    assert read_file_as_string(path) == "hello therewell goodbye"


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(rag, "API_KEY", None)

    with pytest.raises(ValueError):
        rag.Retriever()


def test_embed_or_error_with_mocking():
    with patch("rag.embed_texts") as mock_function:
        mock_function.side_effect = verr.AuthenticationError("test")
        retriever = Retriever()
        result = retriever.embed_or_error(["text"], "document")

    assert result == "There has been an issue with the API key."


def test_index_document_file_not_found(tmp_path):
    retriever = Retriever()
    nonexistent_path = os.path.join(tmp_path, "directory", "file.txt")

    assert retriever.index_document(nonexistent_path) == "File not found."


def test_index_document_invalid_encoding(tmp_path):
    retriever = Retriever()
    path = os.path.join(tmp_path, "file.txt")

    with open(path, "wb") as f:
        f.write(b"\xff")

    assert retriever.index_document(path) == "The file is not UTF-8 encoded."


def test_index_document_exceeds_max_size(tmp_path):
    retriever = Retriever()
    text_chunks = "\n\n".join(f"p{i}" for i in range(301))
    path = os.path.join(tmp_path, "large.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text_chunks)

    assert retriever.index_document(path) == "The document is too large to index."


def test_index_document_embedding_error(tmp_path):
    retriever = Retriever()
    path = os.path.join(tmp_path, "file.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("valid file")

    with patch("rag.embed_texts") as mock_function:
        mock_function.side_effect = verr.AuthenticationError("test")
        result = retriever.index_document(path)

    assert result == "There has been an issue with the API key."
    assert retriever.chunks == []
    assert retriever.embeddings == []


def test_index_document_success(tmp_path):
    retriever = Retriever()
    path = os.path.join(tmp_path, "file.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("chunk 1 \n\n chunk 2")

    with patch("rag.embed_texts") as mock_function:
        mock_function.return_value = [[0.1, 0.2], [0.1, 0.2]]
        result = retriever.index_document(path)

    assert result is None
    assert retriever.chunks == ["chunk 1", "chunk 2"]
    assert retriever.embeddings == [[0.1, 0.2], [0.1, 0.2]]


def test_retrieval_error():
    retriever = Retriever()

    with patch("rag.embed_texts") as mock_function:
        mock_function.side_effect = verr.RateLimitError()
        result = retriever.retrieve("query")

    assert result == "We apologise as there's too many requests at the minute."


def test_retrieval_order():
    retriever = Retriever()
    retriever.chunks = ["medium relevant", "least relevant", "most relevant"]
    retriever.embeddings = [[0.5, 0.5], [-1, 0], [1, 0]]

    with patch("rag.embed_texts") as mock_function:
        mock_function.return_value = [[1, 0]]
        result = retriever.retrieve("query")

    assert result == ["most relevant", "medium relevant", "least relevant"]


def test_retrieval_more_than_k_chunks():
    retriever = Retriever()
    retriever.chunks = ["a", "b", "c", "d"]
    retriever.embeddings = [[1, 0], [0.5, 0.5], [0.3, 0.3], [-1, 0]]

    with patch("rag.embed_texts") as mock_function:
        mock_function.return_value = [[1, 0]]
        result = retriever.retrieve("query")

    assert result == ["a", "b", "c"]
