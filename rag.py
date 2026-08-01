import os
import voyageai
import numpy

from dotenv import load_dotenv

load_dotenv()

# Access the API key stored as an environment variable
API_KEY = os.getenv("VOYAGE_API_KEY")

MODEL = "voyage-4"
TOP_K = 3


class Retriever:
    def __init__(self) -> None:
        if API_KEY is None:
            raise ValueError("No API key.")

        self._client = voyageai.Client(api_key=API_KEY)
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] | list[list[int]] = []

    def index_document(self, path: str) -> None | str:
        try:
            text_string = read_file_as_string(path)
        except FileNotFoundError:
            return "File not found."
        except PermissionError:
            return "You don't have permission for that."
        except UnicodeDecodeError:
            return "The file does not contain UTF-8, UTF-16 or UTF-32 encoded data."
        except OSError as e:
            return f"Something went wrong with the operation: {e}"

        text_chunks = split_text_into_chunks(text_string)
        embeddings = self.embed_or_error(text_chunks, "document")

        if isinstance(embeddings, str):
            return embeddings

        self.chunks = text_chunks
        self.embeddings = embeddings
        return None

    def retrieve(self, query: str) -> list[str] | str:
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
        try:
            return embed_texts(texts, self._client, input_type)
        except voyageai.error.VoyageError as e:
            return explain_api_error(e)


def read_file_as_string(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        file_content = f.read()

    return file_content


def split_text_into_chunks(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    return paragraphs


def embed_texts(
    texts: list[str], client: voyageai.Client, input_type: str
) -> list[list[float]] | list[list[int]]:
    result = client.embed(texts, model=MODEL, input_type=input_type)

    return result.embeddings


def explain_api_error(error: voyageai.error.VoyageError) -> str:
    if isinstance(error, voyageai.error.APIConnectionError):
        return "We apologise as the server could not be reached."
    elif isinstance(error, voyageai.error.AuthenticationError):
        return "There has been an issue with the API key."
    elif isinstance(error, voyageai.error.RateLimitError):
        return "We apologise as there's too many requests at the minute."

    return f"Something went wrong: {error}"
