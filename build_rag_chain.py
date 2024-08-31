from create_custom_retriver_filter_passthrough import RetrievalQAFilter, VectorStoreRetrieverFilter
from mlflow.models import infer_signature
import mlflow
import langchain
import pandas as pd
import time
from mlflow.client import MlflowClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks


# LOAD YOUR SECRETS HERE!!!!
w = WorkspaceClient()

scope = "YOUR_SECRETS"

# w.secrets.delete_secret(scope=scope, key="DATABRICKS_HOST")
# w.secrets.delete_secret(scope=scope, key="DATABRICKS_TOKEN")

# w.secrets.put_secret(scope=scope, key="DATABRICKS_HOST", string_value="foo")
# w.secrets.put_secret(scope=scope, key="DATABRICKS_TOKEN", string_value="bar")
w.secrets.list_secrets(scope=scope)

os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.getBytes(
    scope=scope, key="DATABRICKS_TOKEN").decode("utf-8")
os.environ["DATABRICKS_HOST"] = dbutils.secrets.getBytes(
    scope=scope, key="DATABRICKS_HOST").decode("utf-8")

# LOAD EMBEDDING AND CHAT MODELS
embedding_model_name = "databricks-bge-large-en"
embedding_model = DatabricksEmbeddings(endpoint=embedding_model_name)

# dvs = DatabricksVectorSearch(
#     index, text_column="CHUNK", embedding=embedding_model, columns=["ID", "ID_UNIQUE", "FILE_NAME", "TIMESTAMP"]
# )

# # Test Databricks Foundation LLM model

foundation_model_name = "databricks-meta-llama-3-1-405b-instruct"
# openai_model = "openai-4omni "
chat_model = ChatDatabricks(
    endpoint=foundation_model_name, temperature=0.5, max_tokens=2000)

# LOAD THE CUSTOM CREATED CLASS

# insert prompts and questions
PROMPT = """
YOU DON'T ANSWER, SAY MEOW
""""

questions = ["What is RAG?", "Who am I?"]


def get_retriever_filter(persist_dir: str = None):
    token = os.environ["DATABRICKS_TOKEN"]
    host = os.environ["DATABRICKS_HOST"]
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
    index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=index_name
    )
    # Adjust text_column that contains chunk based on metadata
    vectorstore = DatabricksVectorSearch(
        index, text_column="CHUNK", embedding=embedding_model, columns=["FILE_NAME"]
    )
    return vectorstore


with mlflow.start_run():

    prompt = PromptTemplate(template=PROMPT, input_variables=[
                            "context", "question"])

    # This is to instantiate the VS and can later be overwritten in filter_custom
    search_spec = {"k": 3, "score_threshold": 0.5}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="langchain_core.callbacks.manager")
        retriever_custom = VectorStoreRetrieverFilter(vectorstore=get_retriever_filter(),
                                                      search_type="similarity_score_threshold",
                                                      search_kwargs=search_spec
                                                      )

        # retriever_custom._get_relevant_documents()
        qa = RetrievalQAFilter.from_chain_type(
            llm=chat_model,
            chain_type="stuff",  # TO CHECK
            retriever=retriever_custom,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            # verbose=True
        )
        filter_custom = {"k": 20, "score_threshold": 0.5, "filters": {
            "filter1": ["condition1", "condition2"], "filter2": "condition"}}

        answers = []
        file_names = []
        times = []
        for ii in range(len(questions)):
            start_time = time.time()
            query = questions[ii]
            question_filtered = {"query": query,
                                 "search_kwargs": filter_custom}
            answer = qa(question_filtered)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            answers.append(answer)

        # COLLECT RESULTS
        results = [answers[i]["result"] for i in range(len(answers))]
        source_documents = [answers[i]["source_documents"]
                            for i in range(len(answers))]

        result_df = pd.DataFrame([results, times, source_documents]).T

        # Add a new column called 'prompt' with a static value
        result_df["prompt"] = PROMPT
        result_df["model"] = foundation_model_name
        result_df["embedding_model"] = embedding_model_name
        result_df["questions"] = questions
        result_df["search_config"] = str(filter_custom)

        result_df.columns = ["results", "times", "source_documents",
                             "prompt", "model", "embedding_model", "questions", "search_config"]

    mlflow.log_table(data=result_df, artifact_file="prompt_eng_results.json")

# REGISTER MODEL


def load_filter_retriever(persist_dir: str = None):
    return get_retriever_filter().as_retriever()


source_catalog = "catalog"
source_schema = "schema"

mlflow.set_registry_uri("databricks-uc")
model_name = f"{source_catalog}.{source_schema}.modelname"

with mlflow.start_run(run_name="all_studies_combined") as run:
    # signature = infer_signature(question_filtered, answer)
    model_info = mlflow.langchain.log_model(
        qa,
        # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        loader_fn=load_filter_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question_filtered,
        metadata={"prompt": prompt.template}
        # signature=signature,
    )


# LOAD MODEL IN NEW NOTEBOOK AND TEST FOR INFERENCE

client = MlflowClient()
model_metadata = client.get_latest_versions(model_name, stages=["None"])
latest_model_version = model_metadata[0].version
latest_model_version


# Load the model from the MLflow registry
model_uri = model_metadata[0].source
qa_model = mlflow.langchain.load_model(model_uri)
print(f"Loading model: {model_name} with version {latest_model_version}")


filter_custom = {"k": 20, "score_threshold": 0.5, "filters": {
    "filter1": ["condition1", "condition2"], "filter2": "condition"}}
question_filtered = {"query": "who am I?", "search_kwargs": filter_custom}
answer = qa_model(question_filtered)
