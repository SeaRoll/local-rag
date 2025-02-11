import argparse
from langchain_chroma import Chroma
from documents import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


PROMPT_TEMPLATE = """
다음 맥락만을 바탕으로 질문에 한국어로 답하세요:
{context}

---
위의 맥락을 바탕으로 질문에 한국어로 답하세요: {question}
"""

def query_rag(chroma_path: str, query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="exaone3.5:7.8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

# call like: python prompt.py "이 문서 대해서 요약해줘" 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    print(f"Querying RAG with: {query_text}")
    query_rag("chroma", query_text)
