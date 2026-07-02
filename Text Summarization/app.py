import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )


generic_url = st.text_input(
    "Enter YouTube or Website URL"
)


prompt_template = """
Provide a summary of the following content in 300 words.

Content:
{text}

"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


if st.button("Summarize Content"):

    if not groq_api_key.strip() or not generic_url.strip():

        st.error("Please provide Groq API key and URL")


    elif not validators.url(generic_url):

        st.error("Please enter a valid URL")


    else:

        try:

            with st.spinner("Processing..."):


                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_api_key
                )


                # YouTube loader
                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url
                    )


                # Website loader
                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0"
                        }
                    )


                docs = loader.load()


                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )


                summary = chain.run(docs)


                st.success(summary)


        except Exception as e:

            st.error(f"Error: {e}")