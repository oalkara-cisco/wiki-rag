#!/usr/bin/env python
# coding: utf-8

# # Piecing it all together

# In[ ]:


#!/usr/bin/env python3

import random
import boto3
import chromadb
import streamlit as st
from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import AmazonBedrockEmbeddingFunction
from langchain_community.embeddings import BedrockEmbeddings
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL_ID="amazon.titan-embed-text-v2:0"
MODELS = {
            "llama3-2-3b": "us.meta.llama3-2-3b-instruct-v1:0",
            "llama3-2-11b": "us.meta.llama3-2-11b-instruct-v1:0",
            "llama3-2-90b": "us.meta.llama3-2-90b-instruct-v1:0",
            "llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0"
         }


session = boto3.Session(profile_name='tap_dev')
#session = boto3.Session( region_name="us-east-1")
bedrock_rt = session.client("bedrock-runtime",
                            region_name="us-east-1",
                            )

def get_embeddings():

    embedding_model_id = "amazon.titan-embed-text-v2:0"
    return BedrockEmbeddings(client=bedrock_rt, model_id=embedding_model_id)


def get_vector_collection() -> chromadb.Collection:

    chroma_client = chromadb.PersistentClient(path="./brew-rag-bedrock")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=AmazonBedrockEmbeddingFunction(

                model_name=EMBEDDING_MODEL_ID,
                session=session
            ),
        metadata={"hnsw:space": "cosine"})

def query_collection(prompt: str, n_results: int = 10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str, model_name: str = "llama3-2-3b", temp: float = 0.5):
    system_prompt=f"""
    You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

    context will be passed as within the "documents" tags below:
    <documents>
    {context}
    </documents>

    Please generate 5 follow up questions based on the given context and add it at the end of the response. Do not generate duplicate blocks of follow up questions.

    To answer the question:
    1. Thoroughly analyze the context, identifying key information relevant to the question.
    2. Organize your thoughts and plan your response to ensure a logical flow of information.
    3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
    4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
    5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

    Format your response as follows:
    1. Use clear, concise language.
    2. Organize your answer into paragraphs for readability.
    3. Use bullet points or numbered lists where appropriate to break down complex information.
    4. If relevant, include any headings or subheadings to structure your response.
    5. Ensure proper grammar, punctuation, and spelling throughout your answer.

    Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.

    """
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }
    ]
    system_pt = [
        {
            "text": system_prompt
        }
    ]

    # Configuration for the guardrail.
    guardrail_config = {
        "guardrailIdentifier": "smjijtclxjfc",
        "guardrailVersion": "DRAFT",
        "trace": "enabled"
    }

    model_id = MODELS.get(model_name)
    streaming_response = bedrock_rt.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system_pt,
        inferenceConfig={"temperature": temp, "topP": 0.9},
        guardrailConfig=guardrail_config
    )
    # Extract and print the streamed response text in real-time.
    for chunk in streaming_response["stream"]:
        if "contentBlockDelta" in chunk:
            text = chunk["contentBlockDelta"]["delta"]["text"]
            yield text

def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=5)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == '__main__':
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="Chat with an AI")
        selected_model = st.selectbox(
            'Select a model',
            options=MODELS.keys(),
            index=0
            )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5)

        st.header("Don't know what to ask?")
        process = st.button(
            "⚡️ Generate random questions for me",
        )
        if process:
            collection = get_vector_collection()
            all_documents = collection.get()["documents"]
            sample_docs = random.sample(all_documents, 5)
            prompt="Generate 5 questions that can be asked based on the given context"
            response = call_llm(context=sample_docs, prompt=prompt)
            st.write(response)

    st.header("Chat with an AI")
    prompt = st.text_area("🗣️ Ask a question")
    ask = st.button(
         "🔥 Ask",
    )

    if ask and prompt:
        with st.spinner("Generating response..."):
            results = query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
            response = call_llm(context=relevant_text,
                            prompt=prompt,
                            model_name=selected_model,
                            temp=temperature
                            )
            st.write_stream(response)

