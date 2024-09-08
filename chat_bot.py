import json
import os
import traceback
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import  JsonSpec


app = Flask(__name__)
absolute_path = None # This is used to later delete the chat document

def set_open_api_key():
    """ Method validates and sets OPENAPI key
    """
    with open('config.json', 'r') as file:
        config = json.load(file)
        openai_api_key = config.get('OPENAI_API_KEY')

    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        return True
    return False


def load_document(file):
    """ Method load the documents based on the type of document PDF/JSON
    """
    flag = False
    save_dir = '.'
    file_path = os.path.join(save_dir, file.filename)
    file.save(file_path)
    global absolute_path
    absolute_path = os.path.abspath(file_path)
    if file.mimetype == 'application/pdf':
        loader = PyPDFLoader(absolute_path)
        documents = loader.load()
    elif file.mimetype == 'application/json':
        flag = True
        with open(absolute_path,"r") as f1:
            documents=json.load(f1)
            f1.close()    
    else:
        raise ValueError("Unsupported chat document. Only PDF and JSON are supported.")
    
    return documents, {"json": flag}

def validate_question_file(file):
    """ Method validates questions json file
    """
    if file.mimetype == 'application/json':
        try:
            questions_data = file.read()
            questions_json = json.loads(questions_data.decode('utf-8'))
            return questions_json
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {e}")
    else:
        raise ValueError("Unsupported questions document. Only JSON format is supported.")
    
def validate_file_extension(questions_file, chat_document_file):
    """
    """
    if not questions_file:
        raise Exception("Question file is not passed")
    elif not chat_document_file:
        raise Exception("Chat document is not passed")

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def get_vector_store(docs):
    """ Method creates embedding vector store
    """
    vector_store = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    return vector_store

def get_text_chunks(docs):
    """ Method creates test chuncks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def get_retrieval_chain():
    """ Method creates conversational chain by using OPENAI LLM model
    """
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def answer_questions_pdf(docs, questions):
    """ Main method answers based on the provided questions
    """
    docs = get_text_chunks(docs)
    vector_store = get_vector_store(docs)
    qa_chain=  get_retrieval_chain()
    qa_pairs = []
    for question in questions:
        docs = vector_store.similarity_search(question, k=3)
        result = qa_chain.run(input_documents=docs, question=question)
        qa_pairs.append({question: result})
    return qa_pairs

def answer_questions_json(docs, questions):
    """ Main method answers based on the provided questions
    """
    spec=JsonSpec(dict_=docs, max_value_length=4000)
    toolkit=JsonToolkit(spec=spec)
    agent=create_json_agent(llm=ChatOpenAI(temperature=0,model="gpt-4"), 
                            toolkit=toolkit, max_iterations=1000,verbose=True)
    qa_pairs = []
    for question in questions:
        result = agent.run(question)
        qa_pairs.append({question: result})
    return qa_pairs

@app.route('/QuestionAnswer', methods=['POST'])
def get_answers():
    """ Exposed Flask API to get the answers
    """
    try:
        # Get the uploaded files
        questions_file = request.files.get('questions_file')
        chat_document_file = request.files.get('chat_document_file')

        validate_file_extension(questions_file, chat_document_file)
        questions_json = validate_question_file(questions_file)
        questions = [q['question'] for q in questions_json]

        if not set_open_api_key():
            return jsonify({"error": "OPENAI_API_KEY not found"}), 400

        # Load the document using Langchain loaders
        try:
            documents, flag_dict = load_document(chat_document_file)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if flag_dict["json"]:
            qa_pairs = answer_questions_json(documents, questions)
        else:
            qa_pairs = answer_questions_pdf(documents, questions)
        if absolute_path:
            delete_file(absolute_path)
        return jsonify(qa_pairs)
    except Exception as ex:
        # printing traceback on console
        trc = traceback.format_exc()
        print(trc)
        return jsonify({"error": str(ex)}), 400

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
