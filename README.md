# Socratic Study Assistant
#### Video Demo: https://www.youtube.com/watch?v=RXAHKDcYGXQ
#### Description:
This is a command-line AI chatbot implemented using the Anthropic API for the Large Language Model, with a Streamlit Web UI as a second more user-friendly interface.
It has a Socratic tutor persona made possible with an appropriate system prompt.
Its job is to not just answer questions regarding the user's studies but instead
guide them towards the answer.
In addition, there are some commands (or equivalently buttons in the Web UI) that the user can use, including one for making it give a direct answer.
As a nice addition, I decided to implement Retrieval-Augmented Generation so that the LLM can use context from documents you load in to answer the user's questions or make it generate a quiz on the material.
This is my submission for the CS50P final project, and although the requirements are relatively minimal, I decided to build something more ambitious.

## Features

- **Command-line chat loop**: Loop for alternating between the user's prompt and the assistant's response continuously until termination via `/quit` (via CLI only)
- **Socratic-tutor persona**: System-prompt driven LLM for guiding towards answers rather than handing them over
- **Command-line commands**:
  - `/quit`: to terminate the program and stop talking to the chatbot
  - `/help`: to see the printed list of available commands
  - `/reset`: to clear the conversation history and start fresh
  - `/quiz`: to generate a quiz based on the discussion thus far
  - `/explain`: to make the chatbot directly explain/answer your most recent question
  - `/summarise`: to generate a summary of the discussion thus far
  - `/save [filename]`: to save the conversation to conversations/[filename] in JSON format
  - `/load [path]`: to load a previously saved conversation from given path (JSON only)
  - `/loaddoc [path]`: to load a document from given path so questions can be answered based on it
- **Retrieval-Augmented Generation**: User can load in a document, which gets chunked, embedded and reranked, so answers and quizzes can pull from its content. In the CLI, any non-PDF file is read as plain text regardless of extension; the Streamlit uploader restricts the picker to `.txt`, `.md` and PDF files specifically
- **Streamlit Web UI**: User-friendly web interface mirroring most of the command-line functionality (excluding `/help` and `/quit`, which don't apply to a persistent web page), using sidebar buttons rather than typed commands
- **Streamed responses**: Responses are always streamed in real-time in the CLI. In the Streamlit UI, only the main chat box streams live - the sidebar buttons show the full response at once (deliberate design choice)

## Tech Stack

- **LLM API**: Anthropic API (anthropic SDK)
- **Embeddings and Reranking**: Voyage AI (voyageai SDK)
- **Web UI**: Streamlit
- **Document processing**: pypdf for PDF reading, numpy for the dot-product similarity ranking
- **Config**: python-dotenv
- **Testing**: pytest, fpdf2 for generating PDFs for tests
- **Dev tools**: black, pylint, mypy
- **Documentation**: Sphinx (Furo theme)

## Setup/local run instructions

1. Clone the repo:
   ```
   git clone <repository-url>
   cd python-ai-chatbot
   ```
2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up an `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your-anthropic-key-here
   # VOYAGE_API_KEY is only needed for the RAG feature
   VOYAGE_API_KEY=your-voyage-key-here
   ```
5. Run the CLI:
   ```
   python project.py
   ```
6. (Optional): Run the Streamlit Web UI:
   ```
   streamlit run app.py
   ```
   and visit the outputted URL in your browser

## Design Choices

1. The Socratic persona and the `/explain` command:
   - **Why**: To give it a specific purpose rather than it being a bland LLM: guide the user towards answers with their studies but never answer directly. But this can get annoying so this is resolved via the aforementioned command - the user can request a direct answer when they've made a genuine attempt or just want it.
2. The RAG feature:
   - **Why**: Not just a cool addition but also really useful as a study tool since it lets you quiz/explain/summarise based on your learning material if needed
   - **Why PDF support specifically**: PDF is the most common format my study material uses so I had to make it work with files I actually use
   - **Why manual chunking and similarity ranking**: Voyage's SDK only does embedding and reranking, not chunking or the actual similarity ranking, so I wrote those parts myself with numpy
   - **Dot product, not cosine similarity**: I use a plain dot product (`numpy.dot`) for the ranking, not cosine similarity as such - it only works out the same because Voyage's embeddings are already normalized to length 1, there's no normalization step in my own code
   - **Chunking iteration**: I started with paragraph-based chunking, but paragraph lengths vary too much depending on how a document is formatted, so chunk sizes ended up inconsistent and retrieval suffered - switched to fixed-size overlapping chunks instead for more consistent results
3. The OOP design:
   - The ChatBot class contains the stateful core (message history, client, retriever) and so it earns its place here
   - The Retriever class is also used as it holds state (chunks, embeddings, client)
   - The pure logic in project.py (parse_command, trim_history, is_valid_conversation, etc.) stays as top-level functions - both for testability and because CS50P requires the tested functions to be top-level
4. Isolating the LLM call behind one function:
   - The get_llm_response (for the CLI) and stream_response (for Streamlit) functions are the only places that call the Anthropic API. This makes switching the LLM provider easy as you'd have to change it in one place and keeps the rest of the logic testable without hitting the network
5. Trimming conversation history:
   - **Why**: To keep token usage bounded as conversation grows
   - **Why not just slice naively**: A plain slice to the last N messages could leave an 'orphaned' leading assistant message (or more) with no matching user prompt before it. So the trim loop explicitly trims any leading assistant messages first to keep valid user-assistant pairing before the history gets sent to the API
6. `/save` path validation:
   - **Why**: Found via manual testing that an absolute path or ../ in the filename argument could escape the intended /conversations directory. So in order to enforce one consistent directory to save the conversation files in, I had to check that the user inputs a bare filename only, rejecting anything else
7. `/load` JSON shape validation:
   - **Why**: Found via manual testing that a valid JSON file with the wrong shape i.e. not containing a list of dictionaries with `role` and `content` keys and corresponding string values would get silently accepted by the aforementioned command. So I added a structural check at the point of loading so that bad data never enters the history sent to the API
8. Why the Streamlit buttons don't stream the response:
   - Streamlit reruns the whole script top-to-bottom on every button interaction and the chat-history display loop that renders the messages runs earlier in the script than the buttons are defined, since it has to stay near the top for new messages to render below the existing chat history and not above it. Without a callback, the handle_instruction_button function's changes wouldn't take effect until the next rerun. So handle_instruction_button has to run as a callback i.e. run before the main script body re-executes, so that the state is already updated by the time the display loop runs. If it streamed live from inside that callback, the response would render above the rest of the page - Streamlit's docs say that a callback runs as a 'prefix' to the script, so anything it displays appears before everything else and then disappears once the normal script run takes over and the display loop renders it correctly instead. So instead it 'drains' the stream_response generator method (Streamlit's write_stream accepts a generator or iterable to stream), in project.py, in the background, without displaying each chunk, so that the response is only shown once, in the right place and time.

## Testing

- **Install dev dependencies**: `pip install -r requirements-dev.txt`
- **Run the tests**: `pytest` (from the project root)
- **Covered**:
  - The pure deterministic functions across project.py and rag.py (parsing, history trimming, save/load, JSON shape validation, chunking, document reading, etc.) are tested using plain asserts
  - In addition, there are tests that mock the API-calling functions (get_llm_response, embed_texts) to verify error-handling (rate limit reached, authentication errors, etc.) using `unittest.mock.patch`
  - pytest's `monkeypatch` is used to test the missing API key case
  - `unittest.mock.MagicMock` is used to test that reranking works by constructing a fake response object with the right shape and setting that as the mocked rerank function's return value

## Project Structure

```
python-ai-chatbot/
├── project.py              # CLI entry point: main(), ChatBot class and the required top-level pure functions
├── rag.py                  # RAG: document loading/chunking, embedding, retrieval and reranking (Retriever class)
├── app.py                  # Streamlit web UI, reuses project.py's ChatBot class
├── test_project.py         # pytest tests for project.py
├── test_rag.py             # pytest tests for rag.py
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev dependencies (black, pylint, mypy, sphinx, furo)
├── .env                    # API keys -- create locally, git-ignored
├── conversations/          # Saved conversation JSON files (git-ignored)
└── docs/                   # Sphinx-generated API documentation (HTML + PDF)
```

## Documentation

[Full documentation (PDF)](docs/python-ai-chatbot.pdf)

Interactive HTML docs are also available at [`docs/build/html/index.html`](docs/build/html/index.html) in the repo (clone the repo and open the file locally to view properly)
