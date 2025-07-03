import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


st.title("Video QA Chatbot")

video_url = st.text_input("Enter YouTube Video URL")
lang = st.selectbox("Select Language", ["en", "hi", "fr", "es", "de", "it", "ja", "ko", "pt"])
question = st.text_input("Ask a question about the video")

def extract_youtube_id(url: str) -> str:
    parsed_url = urlparse(url)

    # Handle youtu.be short links
    if parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip('/')

    # Handle youtube.com links
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]

    return None

if st.button("Generate Answer"):
    

    video_id = extract_youtube_id(video_url)
    if video_id:
        try:
           
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
            transcript = None
    else:
        st.error("Could not extract video ID.")
        transcript = None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    if transcript:
        chunks = splitter.create_documents([transcript])
    else:
        st.stop()


    embeddings = HuggingFaceEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question'])

    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    api_key = st.secrets["HUGGINGFACE_API_KEY"]
    llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation",huggingfacehub_api_token=api_key)
    model = ChatHuggingFace(llm=llm)

    answer = model.invoke(final_prompt)
    st.write(answer.content)

else:
    st.info("Please enter a video URL and your question above.")