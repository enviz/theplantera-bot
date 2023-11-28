import streamlit as st
from streamlit_chat import message
import os
import pickle
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from module.chatbot import get_response


def main():
    # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Your own PlantEra bot 🤖")

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            #print(HumanMessage(content=user_input).content)
            st.session_state.messages.append(HumanMessage(content=user_input).content)
            
            with st.spinner("Thinking..."):
                response = get_response(st.session_state.messages[-1])
            st.session_state.messages.append(AIMessage(content=response))
                

    # display message history
    messages = st.session_state.get('messages', [])
    #print(messages)
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    main()


