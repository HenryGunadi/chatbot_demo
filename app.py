"""Simple Streamlit UI. Run with: streamlit run app.py"""
import streamlit as st

from chatbot import Chatbot
from config import Settings

st.set_page_config(page_title="Gemini Chatbot Prototype")
st.title("Gemini Chatbot Prototype")

# Create the chatbot once per session
if "bot" not in st.session_state:
    try:
        st.session_state.bot = Chatbot(Settings.from_env())
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.session_state.messages = []

bot: Chatbot = st.session_state.bot

if st.sidebar.button("Reset conversation"):
    bot.reset()
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("result"):
            r = msg["result"]
            with st.expander("See what happened behind the scenes"):
                st.write(f"**1. Embedding:** {r.embedding_size} dimensions, first 5 values: {r.embedding_preview}")
                st.write("**2. Retrieved context:**")
                for c in r.retrieved_context:
                    st.write(f"- {c}")
                st.write("**3. Final prompt sent to Gemini:**")
                for m in r.final_prompt:
                    st.code(f"[{m.type}] {m.content}", language="text")

# Handle new input
if question := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = bot.ask(question)
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()
        st.markdown(result.answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "result": result}
    )
    st.rerun()