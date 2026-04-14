"""
Knowledge base query demo script.
Demonstrates how to load a pre-built vector knowledge base and perform queries.

Prerequisite: Knowledge base must be built first via build_kb.py
              (medical_kb_structured/ directory must exist).

Usage:
  python kb_demo.py
"""

import os
from typing import Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ─── Configuration ───────────────────────────────────────────────────────────

OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")   # Set via environment variable
MODEL_NAME        = "gpt-4o"
PERSIST_DIRECTORY = "medical_kb_structured"


# ─── Knowledge Base Client ────────────────────────────────────────────────────

class KnowledgeBaseClient:
    """Encapsulates knowledge base loading and query logic"""

    def __init__(
        self,
        persist_dir: str  = PERSIST_DIRECTORY,
        model_name:  str  = MODEL_NAME,
        top_k:       int  = 3,
    ):
        self.embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

        self.qa_chain = self._build_qa_chain(model_name, top_k)

    def _build_qa_chain(self, model_name: str, top_k: int) -> ConversationalRetrievalChain:
        llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True,
            verbose=False,
        )

    def query(self, question: str) -> Dict:
        """
        Query the knowledge base.

        Returns:
            {
                "answer":  str,
                "sources": [{"category": str, "file_name": str}, ...]
            }
        """
        result  = self.qa_chain.invoke({"question": question, "chat_history": []})
        sources = [
            {
                "category":  doc.metadata.get("category", ""),
                "file_name": doc.metadata.get("file_name", ""),
            }
            for doc in result.get("source_documents", [])
        ]
        return {"answer": result.get("answer", ""), "sources": sources}

    def get_diagnosis_info(self, diagnosis: str) -> str:
        """Retrieve standard features and diagnostic criteria for a given diagnosis (for agent use)"""
        result = self.query(
            f"Please describe the typical features, diagnostic criteria and key manifestations of {diagnosis}."
        )
        return result["answer"]

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Direct similarity search (without LLM), useful for debugging or viewing raw content"""
        docs = self.vector_store.similarity_search(query, k=top_k)
        return [
            {
                "content":   doc.page_content,
                "category":  doc.metadata.get("category", ""),
                "file_name": doc.metadata.get("file_name", ""),
            }
            for doc in docs
        ]


# ─── Demo ─────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What are the main immunohistochemical markers expressed in IDH-wildtype glioblastomas?",
    "What are the essential features of appendix adenocarcinoma?",
    "Please describe the typical features and diagnostic criteria of invasive ductal carcinoma of the breast.",
]

def run_demo():
    print("=== Knowledge Base Query Demo ===\n")
    print("Loading knowledge base...")
    client = KnowledgeBaseClient()
    print(f"Knowledge base loaded, vector store path: {PERSIST_DIRECTORY}\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"{'─' * 60}")
        print(f"Question {i}: {question}")
        result = client.query(question)
        print(f"\nAnswer:\n{result['answer']}")
        if result["sources"]:
            print("\nSource documents:")
            for src in result["sources"]:
                print(f"  - [{src['category']}] {src['file_name']}")
        print()

    # Extra demo: diagnosis info retrieval
    print(f"{'─' * 60}")
    diagnosis = "invasive lobular carcinoma"
    print(f"Diagnosis info retrieval (for agent use): {diagnosis}")
    info = client.get_diagnosis_info(diagnosis)
    print(f"\n{info}\n")

    # Extra demo: raw similarity search (without LLM)
    print(f"{'─' * 60}")
    raw_query = "bone marrow biopsy lymphoma"
    print(f"Raw similarity search (Top-3): {raw_query}")
    hits = client.similarity_search(raw_query, top_k=3)
    for j, hit in enumerate(hits, 1):
        print(f"\n  Hit {j}: [{hit['category']}] {hit['file_name']}")
        print(f"  Content snippet: {hit['content'][:150].strip()}...")


if __name__ == "__main__":
    run_demo()
