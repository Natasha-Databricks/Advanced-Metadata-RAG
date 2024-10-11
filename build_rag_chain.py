# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC %pip install langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import langchain
import pandas as pd
import time
import os


from databricks.vector_search.client import VectorSearchClient
from mlflow.models import infer_signature
from mlflow.client import MlflowClient

from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks
from langchain_core.prompts import PromptTemplate

import warnings
from warnings import filterwarnings
filterwarnings("ignore")

# COMMAND ----------

# THOSE ARE OUR EXTENDED LANGCHAIN MODULES TO PASS THE FILTER ARGUMENTS THROUGH THE CHAIN
from create_custom_retriver_filter_passthrough import RetrievalQAFilter, VectorStoreRetrieverFilter

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()

# scope = "natasha_secrets"
# w.secrets.delete_secret(scope=scope, key="DATABRICKS_TOKEN")
# w.secrets.put_secret(scope=scope, key="DATABRICKS_TOKEN", string_value="foo")
# w.secrets.list_secrets(scope=scope)

os.environ["DATABRICKS_TOKEN"] = ""
os.environ["DATABRICKS_HOST"] = ""

# COMMAND ----------

# LOAD EMBEDDING AND CHAT MODELS
embedding_model_name = "databricks-bge-large-en"
embedding_model = DatabricksEmbeddings(endpoint=embedding_model_name)

# COMMAND ----------


foundation_model_name = "databricks-meta-llama-3-1-405b-instruct"
# openai_model = "openai-4mni "
chat_model = ChatDatabricks(
    endpoint=foundation_model_name, temperature=0.5, max_tokens=2000)


# COMMAND ----------

# insert prompts and questions
def get_retriever_filter(persist_dir: str = None):
    token = os.environ["DATABRICKS_TOKEN"]
    host = os.environ["DATABRICKS_HOST"]
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
    index = vsc.get_index(
        endpoint_name="<>",
        index_name="<>"
    )
    # Adjust text_column that contains chunk based on metadata
    vectorstore = DatabricksVectorSearch(
        index, text_column="content", embedding=embedding_model, columns=["content", "url"]
    )
    return vectorstore

# COMMAND ----------

PROMPT = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer. End your response with "Thank you for your attention ML SMEs

{context}

Human: {question}
Chatbot:"""


questions = ["What are audit logs?", "Why do audit logs matter?"]

# COMMAND ----------

prompt = PromptTemplate(template=PROMPT, input_variables=[
                  "context", "question"])

# This is to instantiate the VS and can later be overwritten in filter_custom
search_spec = {"num_results": 3} #TO-DO: add score threshold which has been buggy

retriever_custom = VectorStoreRetrieverFilter(vectorstore=get_retriever_filter(),
                                            search_type="similarity",
                                            search_kwargs=search_spec
                                            )

# retriever_custom._get_relevant_documents()
qa = RetrievalQAFilter.from_chain_type(
  llm=chat_model,
  chain_type="stuff",  # TO CHECK
  retriever=retriever_custom,
  chain_type_kwargs={"prompt": prompt},
  return_source_documents=True,
  #verbose=True
)

filter_custom = {"num_results": 5, "filters": {
  "url": "https://docs.databricks.com/en/admin/account-settings/audit-logs.html"}}

question_filtered = {
    "query": questions[0],
    "search_kwargs": filter_custom  # Only if search_kwargs is expected
}


result = qa(question_filtered)
print(result)



# COMMAND ----------

import mlflow.pyfunc
import logging
from langchain.prompts import PromptTemplate

class RAGModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, prompt_template, retriever_spec, retriever_class, qa_class):
        self.prompt_template = prompt_template
        self.retriever_spec = retriever_spec
        self.retriever_class = retriever_class
        self.qa_class = qa_class

    def load_context(self, context):
        # Log the prompt template to check its content
        logging.info(f"Prompt Template: {self.prompt_template}")
        assert self.prompt_template, "prompt_template must not be empty."

        # Instantiate retriever and QA filter with necessary parameters
        retriever_custom = self.retriever_class(
            vectorstore=get_retriever_filter(),
            search_type="similarity",
            search_kwargs=self.retriever_spec
        )

        # Initialize the prompt template
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])

        # Initialize QA model
        self.qa_model = self.qa_class.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever_custom,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def predict(self, context, input_data):
        question_filtered = {
            "query": input_data.get("query"),
            "search_kwargs": input_data.get("search_kwargs", {})
        }
        
        result = self.qa_model(question_filtered)
        return result

# Logging Setup
logging.basicConfig(level=logging.INFO)

# Define your configuration
prompt_template = "{context}: {question}"  # Ensure the template string is correctly formatted
retriever_spec = {"num_results": 3}
filter_custom = {
    "num_results": 5,
    "filters": {"url": "https://docs.databricks.com/en/admin/account-settings/audit-logs.html"}
}

# Log the model in MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="rag_pyfunc_model",
        python_model=RAGModelWrapper(
            prompt_template=prompt_template,
            retriever_spec=retriever_spec,
            retriever_class=VectorStoreRetrieverFilter,
            qa_class=RetrievalQAFilter
        )
    )


# COMMAND ----------

import mlflow

# Define the input for inference
question_filtered = {
    'query': 'How to set up IAM credentials',
    'search_kwargs': {
        'num_results': 5,
        'filters': {
            'url': 'https://docs.databricks.com/en/admin/account-settings-e2/credentials.html'
        }
    }
}

# Load the model from MLflow
model_uri = f'runs:/{run.info.run_uuid}/rag_pyfunc_model'
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Run inference
result = loaded_model.predict(question_filtered)

# Print the result
print(result)


# COMMAND ----------

# vectorstore=get_retriever_filter()

# vectorstore.similarity_search_with_relevance_scores(
#     query="what is unity catalog?",
#     columns=["url", "content"],
#     filters={'url': ['https://docs.databricks.com/en/admin/account-settings/audit-logs.html']},
#     num_results=2,
#     threshold=0.5
#     )
