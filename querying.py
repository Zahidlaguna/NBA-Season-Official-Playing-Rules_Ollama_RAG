import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding import get_embedding

CHROMA_PATH = 'chroma'

PROMPT_TEMPLATE = '''
Analyze the document and answer the following questions based only on context:
{context}

answer the question: {question}'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help='Text to query')
    args = parser.parse_args()
    query = args.query_text
    rag_query(query)
    
def rag_query(query: str):
    embedding_function = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query, k=5)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    
    model = Ollama(model='llama3')
    response = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

if __name__ == '__main__':
    main()