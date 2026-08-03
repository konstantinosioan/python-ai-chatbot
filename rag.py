"""Document loading, chunking, embedding and retrieval for the RAG feature using the Voyage AI API"""

import os
import voyageai
import numpy

from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PyPdfError

load_dotenv()

# Access the API key stored as an environment variable
API_KEY = os.getenv("VOYAGE_API_KEY")

MODEL = "voyage-4"
TOP_K = 3
MAX_CHUNKS = 300
CHUNK_SIZE_WORDS = 175
CHUNK_OVERLAP_WORDS = 40

# Caps uploaded pdf size at 20MB
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024


class Retriever:
    """Holds an indexed document's chunks and embeddings, and retrieves the most relevant chunks for a given query"""

    def __init__(self) -> None:
        """
        Initialises empty chunks/embeddings lists until a document is loaded and sets up the Voyage AI client

        :raise ValueError: If the API key is missing
        """

        if API_KEY is None:
            raise ValueError("No API key.")

        self._client = voyageai.Client(api_key=API_KEY)
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] | list[list[int]] = []

    def index_document(self, path: str) -> None | str:
        """
        Reads a text or PDF file from a given path, splits it into chunks, embeds them and stores the chunks/embeddings on success. On failure, it returns an error message

        :param path: The path from which to index
        :return: None on success; friendly error message on failure
        """
        try:
            text_string = read_document_as_string(path)
        except FileNotFoundError:
            return "File not found."
        except PermissionError:
            return "You don't have permission for that."
        except UnicodeDecodeError:
            return "The file is not UTF-8 encoded."
        except FileNotDecryptedError:
            return "This PDF is encrypted and could not be read."
        except PyPdfError:
            return "This PDF is corrupted or could not be read."
        except ValueError as e:
            return str(e)
        except OSError as e:
            return f"Something went wrong with the operation: {e}"

        text_chunks = split_text_into_chunks(text_string)

        if len(text_chunks) > MAX_CHUNKS:
            return "The document is too large to index."

        embeddings = self.embed_or_error(text_chunks, "document")

        if isinstance(embeddings, str):
            return embeddings

        self.chunks = text_chunks
        self.embeddings = embeddings
        return None

    def retrieve(self, query: str) -> list[str] | str:
        """
        Embeds the query and returns the top k most relevant indexed chunks, ranked by cosine similarity

        :param query: The text to find relevant chunks for
        :return: A list of most relevant chunks (up to TOP_K) on success; friendly error message on failure
        """
        result = self.embed_or_error([query], "query")

        if isinstance(result, str):
            return result

        embedding = result[0]

        # Voyage embeddings are normalised to length 1, so dot product and cosine similarity are the same
        dot_products = numpy.dot(self.embeddings, embedding)

        # The indices of the elements when the data is sorted by similarity in descending order
        sorted_indices = numpy.argsort(dot_products)[::-1]

        # Take the top k related chunks
        top_k_indices = sorted_indices[:TOP_K]
        top_k_chunks = [self.chunks[i] for i in top_k_indices]

        return top_k_chunks

    def embed_or_error(
        self, texts: list[str], input_type: str
    ) -> list[list[float]] | list[list[int]] | str:
        """
        Embeds the given texts and returns them on success. On failure, it returns an error message

        :param texts: The list of texts to embed
        :param input_type: Either "query" or "document" per Voyage AI's embedding API
        :return: The list of embeddings on success; friendly error message on failure
        """
        try:
            return embed_texts(texts, self._client, input_type)
        except voyageai.error.VoyageError as e:
            return explain_api_error(e)


def read_file_as_string(path: str) -> str:
    """
    Reads a file's content as a UTF-8 encoded string and returns a string containing content

    :param path: The path from which to read file
    :raise FileNotFoundError: If the file cannot be found
    :raise PermissionError: If user does not have the necessary permissions to read from given path
    :raise UnicodeDecodeError: If the file is not UTF-8 encoded
    :raise OSError: If any other thing goes wrong when reading file
    :return: The string containing file content
    """
    with open(path, "r", encoding="utf-8") as f:
        file_content = f.read()

    return file_content


def read_pdf_as_string(path: str) -> str:
    """
    Reads a PDF's content and returns a string containing it

    :param path: The path from which to read the PDF
    :raise FileNotFoundError: If the file cannot be found
    :raise PermissionError: If user does not have the necessary permissions to read from given path
    :raise pypdf.errors.FileNotDecryptedError: If an encrypted PDF has not been successfully decrypted
    :raise pypdf.errors.PyPdfError: If the PDF is corrupted, empty or unreadable
    :raise OSError: If any other thing goes wrong when reading the file
    :return: The string containing the PDF's content
    """
    reader = PdfReader(path)

    pages_text = "\n\n".join(
        page.extract_text(extraction_mode="layout") for page in reader.pages
    )

    return pages_text


def read_document_as_string(path: str) -> str:
    """
    Reads a document's content as a string, detecting whether it's a PDF or a plain text file and acting appropriately

    :param path: The path from which to read the document
    :raise ValueError: If the file is a PDF larger than MAX_PDF_SIZE_BYTES
    :raise FileNotFoundError: If the file cannot be found
    :raise PermissionError: If user does not have the necessary permissions to read from given path
    :raise UnicodeDecodeError: If a plain text file is not UTF-8 encoded
    :raise pypdf.errors.FileNotDecryptedError: If an encrypted PDF has not been successfully decrypted
    :raise pypdf.errors.PyPdfError: If the PDF is corrupted, empty or unreadable
    :raise OSError: If any other thing goes wrong when reading the file
    :return: The string containing the document's content
    """
    with open(path, "rb") as f:
        header = f.read(5)
        is_pdf = header == b"%PDF-"

    if is_pdf:
        if os.path.getsize(path) > MAX_PDF_SIZE_BYTES:
            raise ValueError("The PDF is too large to index.")

        return read_pdf_as_string(path)

    return read_file_as_string(path)


def split_text_into_chunks(text: str) -> list[str]:
    """
    Splits text into CHUNK_SIZE_WORDS-sized chunks. Each chunk overlaps with the next by CHUNK_OVERLAP_WORDS words

    :param text: The text to split
    :return: A list of the overlapping, fixed-size chunks
    """
    # Split the text by whitespace
    words = text.split()

    if not words:
        return []

    chunks = []
    increment = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    for i in range(0, len(words), increment):
        chunk_words = " ".join(words[i : i + CHUNK_SIZE_WORDS])
        chunks.append(chunk_words)

        if i + CHUNK_SIZE_WORDS >= len(words):
            break

    return chunks


def embed_texts(
    texts: list[str], client: voyageai.Client, input_type: str
) -> list[list[float]] | list[list[int]]:
    """
    Sends texts to the Voyage API and returns their embeddings

    :param texts: The list of texts to embed (document chunks or a single query)
    :param client: The Voyage AI client
    :param input_type: Either "query" or "document" per Voyage AI's embedding API
    :raise voyageai.error.APIConnectionError: If any network connection failures occur
    :raise voyageai.error.AuthenticationError: If the API key is invalid, expired or revoked
    :raise voyageai.error.RateLimitError: If the rate limit is exceeded
    :raise voyageai.error.VoyageError: If anything other than the above goes wrong regarding the API
    :return: The list of embeddings, one per input text
    """
    result = client.embed(texts, model=MODEL, input_type=input_type)

    return result.embeddings


def explain_api_error(error: voyageai.error.VoyageError) -> str:
    """
    Returns a friendly error message for each caught Voyage AI exception

    :param error: The general VoyageError caught
    :return: A friendly error message
    """
    if isinstance(error, voyageai.error.APIConnectionError):
        return "We apologise as the server could not be reached."
    elif isinstance(error, voyageai.error.AuthenticationError):
        return "There has been an issue with the API key."
    elif isinstance(error, voyageai.error.RateLimitError):
        return "We apologise as there's too many requests at the minute."

    return f"Something went wrong: {error}"


def augment_system_prompt(system_prompt: str, chunks: list[str]) -> str:
    """
    Builds a system prompt with additional context by appending the retrieved file chunks and instructing the model to use them and say when the answer isn't contained in them

    :param system_prompt: The base system prompt
    :param chunks: The list of retrieved chunks to ground the prompt in
    :return: The system prompt extended with the context
    """
    context_block = (
        "Here is relevant context retrieved from the user's loaded document:"
        + "\n\n"
        + "\n\n".join(chunks)
        + "\n\n"
        + "Only use the context above to answer if relevant. If the answer is not contained in it, do say so."
        + "\n\n"
        + "Note that the context is genuinely available to you. Even if you said earlier in this conversation that you couldn't access files, that's no longer true. Treat the above content as real."
    )

    return system_prompt + "\n\n" + context_block
