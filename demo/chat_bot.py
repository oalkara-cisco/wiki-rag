#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import streamlit as st
import uuid


def bedrock_chain():
    session = boto3.Session(profile_name='tap_dev')

    # Create a bedrock runtime client in us-east-1
    bedrock_runtime = session.client(
        "bedrock-runtime",
        region_name="us-east-1"
    )

    titan_llm = Bedrock(
        model_id="us.meta.llama3-2-3b-instruct-v1:0", client=bedrock_runtime, credentials_profile_name="tap_dev", provider="meta"
    )
    # "prompt": "{prompt}" # The prompt variable from Langchain is placed here.
    titan_llm.model_kwargs = {
        "temperature": 0.5,

    }

    prompt_template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
    The assistant provides short answers from its context unless asked for a detailed answer.

    Current conversation:
    {history}

    Human: {input}
    Assistant:"""

    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template

    )

    template = """Instruction: You are a chatbot that is unhelpful. Your goal is to not help the user but only make jokes. 
    Take what the user is saying and make a joke out of it.

    {chat_history}
    Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)

    memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=titan_llm,
        verbose=True,
        memory=memory,
    )

    return conversation

def run_chain(chain, prompt):
    num_tokens = chain.llm.get_num_tokens(prompt)
    return chain({"input": prompt}), num_tokens


def clear_memory(chain):
    return chain.memory.clear()

def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 3])
    with col2:
        header = "Amazon Bedrock Demo Chatbot"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear


def handle_input():
    input = st.session_state.input

    llm_chain = st.session_state["llm_chain"]
    #chain = st.session_state["llm_app"]
    result, amount_of_tokens = run_chain(llm_chain, input)
    question_with_id = {
        "question": input,
        "id": len(st.session_state.questions),
        "tokens": amount_of_tokens,
    }
    st.session_state.questions.append(question_with_id)

    st.session_state.answers.append(
        {"answer": result, "id": len(st.session_state.questions)}
    )
    st.session_state.input = ""


def write_user_message(md):
    col1, col2 = st.columns([1, 12])

    with col1:
        st.image(USER_ICON, use_container_width="always")
    with col2:
        st.warning(md["question"])


def render_answer(answer):
    col1, col2 = st.columns([1, 12])
    with col1:
        st.image(AI_ICON, use_container_width="always")
    with col2:
        st.info(answer["response"])


def write_chat_message(md):
    chat = st.container()
    with chat:
        render_answer(md["answer"])


USER_ICON = "https://t3.ftcdn.net/jpg/03/94/89/90/360_F_394899054_4TMgw6eiMYUfozaZU3Kgr5e0LdH4ZrsU.jpg"
AI_ICON = "https://static.vecteezy.com/system/resources/thumbnails/009/971/218/small/chat-bot-icon-isolated-contour-symbol-illustration-vector.jpg"

if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
else:
    user_id = str(uuid.uuid4())
    st.session_state["user_id"] = user_id

if "llm_chain" not in st.session_state:
    st.session_state["llm_app"] = "bedrock"
    st.session_state["llm_chain"] = bedrock_chain()

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input" not in st.session_state:
    st.session_state.input = ""

clear = write_top_bar()

if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
    clear_memory(st.session_state["llm_chain"])


with st.container():
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        write_user_message(q)
        write_chat_message(a)


st.markdown("---")
input = st.text_input(
    "You are talking to an AI, ask any question.", key="input", on_change=handle_input
)


# In[ ]:




