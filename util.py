
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([chunk["text"] for chunk in transcript])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        st.error(f"❌ Transcript error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def embed_chunks(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embeddings)
