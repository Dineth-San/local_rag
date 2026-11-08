import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Local RAG with Llama 3.2 & Qdrant")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Upload a document, then ask me anything about it!")

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = ""
if "current_file" not in st.session_state:
    st.session_state.current_file = ""

uploaded_file = st.file_uploader("Uplaod a PDF to analyze", type="pdf")


if uploaded_file is not None:

    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.doc_summary = ""
        msgs.clear()
        msgs.add_ai_message(f"I've loaded '{uploaded_file.name}'. What would you like to know?")

    # save the file temporarily so pypdfloader can read it
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    if not st.session_state.doc_summary:
        with st.spinner("Processing PDF and sending vectors to Qdrant..."):
            # 2. Extract and split text
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            
            # 3. Embed and store in Qdrant 
            vectorstore = QdrantVectorStore.from_documents(
                splits,
                embeddings,
                url="http://localhost:6333",
                collection_name="local_knowledge",
                force_recreate=True, 
            )

            intro_text = " ".join([doc.page_content for doc in splits[:3]])
            summary_prompt = f"Based on t his introductor text, summarize what this entire document is about in 3 concise sentences. Text: {intro_text}"
            st.session_state.doc_summary = llm.invoke(summary_prompt).content

            st.success(f"Document processed!")


    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="local_knowledge"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    with st.expander("View document overview (macro-context)"):
        st.write(st.session_state.doc_summary)

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if question := st.chat_input("Ask a question based on the document: "):
        st.chat_message("human").write(question)

        with st.spinner("Llama 3.2 is thinking..."):
            # prompt 1: rewrite the question using chat history so Qdrant can understand
            contextualize_q_system_prompt = (
                "Given a chat history and the latest use question "
                "which might referencce contextt in the chat history. "
                "formulate a standalone question which can be understood without the chat history."
                "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"), # this injects the memory
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # prompt 2: answer the question using the retrieved PDF chunks
            qa_system_prompt = (
                "You are an assistant forr question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                f"Document Overview: {st.session_state.doc_summary}\n"
                "specific context: {context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: msgs,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: msgs,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            response = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "default_session"}}
            )

            st.chat_message("ai").write(response["answer"])
    