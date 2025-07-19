import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql import SQLDatabaseSequentialChain
from langchain_experimental.sql import SQLDatabaseChain

import google.generativeai as genai

# --- Load API Key ---

from secret_key import GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)


if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in secret_key.py.")
    st.stop()

# Configure for Google Generative AI + LangChain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 FMCG Sales Chatbot")
st.title("📊 FMCG Sales Analysis Chatbot")

uploaded_file = st.file_uploader("📁 Upload your retail sales CSV", type=["csv"])

if uploaded_file:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Save to SQLite
    engine = create_engine("sqlite:///retail_sales.db")
    df.to_sql("sales_data", engine, if_exists="replace", index=False)
    st.success("✅ Data loaded into SQLite database.")

    # LangChain DB connection
    db = SQLDatabase.from_uri("sqlite:///retail_sales.db")

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)

    # Generic prompt template for natural language answers
    prompt = PromptTemplate(
        input_variables=["query", "sql_result"],
        template=(
            "You are a helpful assistant that answers questions based on SQL query results.\n"
            "User asked: {query}\n"
            "Here is the SQL result data:\n{sql_result}\n"
            "Please provide a concise, clear, natural language summary of the result without showing SQL or raw data tables."
        )
    )

    db_chain = SQLDatabaseSequentialChain.from_llm(
    llm=llm,
    db=db,
    verbose=False,
    return_intermediate_steps=False
)


    query = st.text_input("Ask a question about your sales data:")

    if query:
        with st.spinner("Thinking..."):
            try:
                response = db_chain.run(query)
                st.markdown(f"💬 **Answer:** {response}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")
