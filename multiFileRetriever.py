import os

from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI()


# loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

persist_directory = 'multifile'
embedding = OpenAIEmbeddings()

# vectordb = Chroma.from_documents(documents=texts,
#                                  embedding=embedding,
#                                  persist_directory=persist_directory)


# vectordb.persist()
# vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

# retriever = vectordb.as_retriever()

# docs = retriever.get_relevant_documents("Why is my appointment not visible")
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

template = """
The following answer was generated by a dumb ai. It may have sentences broken into smaller sentences for no reason
It may also not be displayed in bullet points / numbered list when it should.
Fix that. Pay special attention to whether a bullet points or a list should be used

answer: {answer}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["answer"],
    template=template
)

while True:
    query = input("Enter query:")
    llm_response = qa_chain(query)
    print(llm(prompt_template.format(
        answer=llm_response['result']
    )))

    # process_llm_response(llm_response)



