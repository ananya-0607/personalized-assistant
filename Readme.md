Personalized Learning Assistant (In Progress)
A prototype for asking questions about personal study notes. The intended flow is: load notes, split them into chunks, store embeddings in ChromaDB, retrieve relevant chunks, and send the retrieved text to Gemini to draft an answer.
Current code
- src/ingestion.py reads data/Web Tech Notes.txt and uses CharacterTextSplitter with chunk_size=500 and chunk_overlap=50.
- src/embeddings.py sets up local all-MiniLM-L6-v2 sentence embeddings and ChromaDB storage.
- src/retrieval.py requests the three most similar chunks for a question.
- src/app.py provides a basic Flask form and prompts Gemini to answer from retrieved context, saying "I don't know" when the answer is absent.
- src/qa.py is an older, unused experiment and is not part of the Flask flow.
Status and next steps
The end-to-end application is not yet verified. The current code passes a raw SentenceTransformer object to LangChain's Chroma integration; this needs a compatible embedding wrapper or embedding interface. After fixing that, I plan to:
1. Add requirements.txt and .env.example containing placeholders only.
2. Add sample notes and clear setup commands.
3. Test ingestion, storage, retrieval, and Flask question answering together.
4. Show the supporting source chunks and handle questions that the notes cannot answer.
No API keys or generated vector database files should be committed. Keep the Gemini key in a local .env file and ignore that file in Git.
