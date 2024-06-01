import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
# from langchain.evaluation import load_evaluator
# from langchain.evaluation import EvaluatorType
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You should now take on the role of an AI assistant, extracting key-value pairs from the papers I upload. Since the college is called Prasiddesh Institute of Technology, you should extract the two terms that are listed in the document's boxes: "Name" and "USN". The "USN" is a combination of alphabets and numbers. Row-wise data must be concatenated in order to obtain the "Name" and "USN". Additionally, you must extract the marks that are listed in the document's conclusion.
For the document uploaded you should give the output only in the below mentioned format (No extra explaination should be given):
\n
Name:
\n 
USN:
\n 
Marks: 

Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


# def llm_evaluation():
#     prompt = """
# You should now take on the role of an AI assistant, extracting key-value pairs from the papers I upload. Since the college is called Prasiddesh Institute of Technology, you should extract the two terms that are listed in the document's boxes: "Name" and "USN". The "USN" is a combination of alphabets and numbers. Row-wise data must be concatenated in order to obtain the "Name" and "USN". Additionally, you must extract the marks that are listed in the document's conclusion.
# For the document uploaded you should give the output only in the below mentioned format (No extra explaination should be given):
# \n
# Name:
# \n 
# USN:
# \n 
# Marks: 
#     """
#     evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria='conciseness')
#     model = Ollama(model="mistral")
#     prediction = model.invoke(prompt)
#     eval_result = evaluator.evaluate_strings(prediction=prediction, input=prompt)

#     print("\nPROMPT: ", prompt)
#     print("RESULT: \n", '\n'.join(prediction.replace('\n', '').split('.')[:-1]))
#     print("VALUE: ", eval_result['value'])
#     print("SCORE: ", eval_result['score'])
#     print("REASON: \n", '\n'.join(eval_result['reasoning'].replace('\n', '').split('.')[:-1]))



if __name__ == "__main__":
    main()
    # llm_evaluation()
