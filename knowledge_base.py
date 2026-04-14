import os
from typing import Dict, Optional
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from model_client import create_model_client, OPENAI_BASE_URL as _BASE_URL, OPENAI_API_KEY as _API_KEY


class MedicalKnowledgeBuild:
    """Medical knowledge base builder"""

    # Reuse the same base URL and API key as model_client.py
    OPENAI_BASE_URL = _BASE_URL
    OPENAI_API_KEY  = _API_KEY
    MODEL_NAME = "gpt-4o"

    def __init__(self, base_dir: str, persist_directory: str = None):
        """Initialize medical knowledge base

        Args:
            base_dir:          Path to raw document source directory.
            persist_directory: Path to Chroma vector store. Defaults to
                               Config.KB_DIR when None.
        """
        from config import Config
        self.base_dir = Path(base_dir)

        self.model_client = create_model_client(self.MODEL_NAME)

        self.embeddings = OpenAIEmbeddings(
            api_key=self.OPENAI_API_KEY,
            base_url=self.OPENAI_BASE_URL,
        )

        self.persist_directory = persist_directory or Config.KB_DIR

    def create_qa_chain(self, vector_store: Optional[Chroma] = None) -> ConversationalRetrievalChain:
        """Create QA chain"""
        try:
            if vector_store is None:
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )

            llm = ChatOpenAI(
                temperature=0,
                model_name=self.MODEL_NAME,
                openai_api_key=self.OPENAI_API_KEY,
                openai_api_base=self.OPENAI_BASE_URL,
            )

            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True,
                verbose=False
            )
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            raise

class MedicalKnowledgeBase:
    """Medical knowledge base wrapper for agent use"""

    def __init__(self, qa_chain: ConversationalRetrievalChain):
        """Initialize with existing QA chain"""
        self.qa_chain = qa_chain

    def query(self, question: str) -> Dict:
        """Query knowledge base"""
        try:
            result = self.qa_chain.invoke({"question": question, "chat_history": []})

            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "category": doc.metadata.get("category"),
                    "file_name": doc.metadata.get("file_name")
                })

            return {
                "answer": result.get("answer", ""),
                "sources": sources
            }
        except Exception as e:
            print(f"Error querying knowledge base: {str(e)}")
            raise

    def get_diagnosis_info(self, diagnosis: str) -> str:
        """Get information about a specific diagnosis"""
        try:
            result = self.query(
                f"Please describe the typical features, diagnostic criteria and key manifestations of {diagnosis}."
            )
            return result["answer"]
        except Exception as e:
            print(f"Error getting diagnosis info: {str(e)}")
            raise
