import os

from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI()


template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Ensure that sentences in your answers are not broken and that, when required, a clear bullet points list is displayed.
{context}
Question: {question}
Helpful Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

persist_directory = 'multifile'
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, verbose=True), vectordb.as_retriever(search_kwargs={"k": 5}), memory=memory, combine_docs_chain_kwargs={"prompt": prompt_template})


print()


while True:
    query = input("question: ")
    result = qa({"question": query})
    print(result["answer"])
    # print(llm(prompt_template.format(
    #     answer=result['answer']
    # )))



