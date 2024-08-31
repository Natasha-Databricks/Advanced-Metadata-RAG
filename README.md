# README

## Overview

This project contains two main Python scripts that integrate LangChain with Databricks and MLflow to create a custom Retrieval-Augmented Generation (RAG) chain. The scripts allow for the customization of metadata filters in a Langchain vector search module.

### Scripts

1. **create_custom_retriever_filter_passthrough.py**: 
   - This script defines custom classes that extend the LangChain functionality. The custom classes, `RetrievalQAFilter` and `VectorStoreRetrieverFilter`, enable the passing of custom metadata filters through to the vector store, allowing for more refined and relevant document retrieval based on specific conditions.

2. **build_rag_chain.py**:
   - This script uses the custom classes defined in the first script to build and execute a RAG chain. It sets up the environment, retrieves secrets, configures embedding and chat models, and performs retrieval and question answering. The results are logged and stored using MLflow, and the model is registered for future use. Please note that that there might be compatability issues between MLFlow and the custom Langchain class which will lead to serlialisation issues. In this case, it is recommended to serlaise the code as custom mlflow pyfunc module. 

## Requirements

- Python 3.7+
- Databricks environment with access to Databricks Secret Scopes
- MLflow
- LangChain
- Databricks-specific Python packages for vector search and embeddings

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Advanced-Metadata-RAG.git
    cd Advanced-Metadata-RAG
    ```

2. Install the required Python packages:
   *To-Do*

3. Set up your Databricks environment with the required secrets:
   - Create a secret scope in Databricks and store your Databricks host and token as secrets.

## Usage

### Step 1: Create Custom Retriever and Filter

The `create_custom_retriever_filter_passthrough.py` script defines two main classes:

- `RetrievalQAFilter`: A custom class extending LangChain's `RetrievalQA` that allows for using search keywords (`search_kwargs`) to filter and retrieve relevant documents based on custom metadata.

- `VectorStoreRetrieverFilter`: A custom class that extends `VectorStoreRetriever`, enabling the integration of metadata-based filtering in vector-based search processes.

### Step 2: Build and Run the RAG Chain

1. Configure your Databricks secrets and environment variables in the `build_rag_chain.py` script:

    ```python
    scope = "YOUR_SECRETS_SCOPE_NAME"
    os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.getBytes(scope=scope, key="DATABRICKS_TOKEN").decode("utf-8")
    os.environ["DATABRICKS_HOST"] = dbutils.secrets.getBytes(scope=scope, key="DATABRICKS_HOST").decode("utf-8")
    ```

2. Customize the embedding and chat model parameters:
   
   - Set the `embedding_model_name` and `foundation_model_name` to your specific Databricks model endpoints.

3. Define the prompts and questions:
   - Modify the `PROMPT` variable and `questions` list to customize the input for the question-answering task.

4. Execute the `build_rag_chain.py` script:
    ```bash
    python build_rag_chain.py
    ```

   This script will:
   - Initialize the retriever with custom filters.
   - Run the QA chain with the specified questions.
   - Log the results using MLflow.
   - Register the trained model for future use.

### Step 3: Model Registration and Inference

- After running the script, the model will be registered in the MLflow registry.
- Use the provided MLflow client code to load the latest model version and perform inference.

```python
client = MlflowClient()
model_metadata = client.get_latest_versions(model_name, stages=["None"])
latest_model_version = model_metadata[0].version
qa_model = mlflow.langchain.load_model(model_metadata[0].source)
