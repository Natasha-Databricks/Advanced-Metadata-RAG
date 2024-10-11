import numpy as np
import inspect
import os

from typing import Any, List, Dict, Union, Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    CallbackManagerForRetrieverRun,
    Callbacks,
    AsyncCallbackManagerForRetrieverRun,
)

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.vectorstores.utils import maximal_marginal_relevance


class RetrievalQAFilter(RetrievalQA):
    '''Custom RetrievalQA Chain with filter functionality.

    This class extends the standard RetrievalQA chain to include additional filtering capabilities. It overrides
    methods to provide support for filtering documents based on custom search keyword arguments (`search_kwargs`)
    and manages callbacks using a callback manager for both synchronous and asynchronous operations.
    '''

    def _get_docs(
        self,
        question: str,
        search_kwargs: Dict[str, Any] = None,
        *,
        run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        '''Retrieve relevant documents for a given question.

        Args:
            question (str): The question for which to retrieve relevant documents.
            search_kwargs (Dict[str, Any], optional): Additional keyword arguments for the search.
            run_manager (CallbackManagerForChainRun): The callback manager to handle run callbacks.

        Returns:
            List[Document]: A list of relevant documents retrieved based on the provided question.
        '''
        # Check if search_kwargs is provided
        if search_kwargs is not None:
            docs = self.retriever._get_relevant_documents(
                question, search_kwargs=search_kwargs, run_manager=run_manager)
            return docs
        else:
            return self.retriever._get_relevant_documents(question)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: CallbackManagerForChainRun = None
    ) -> Dict[str, Any]:
        '''Process input and generate an answer using the retrieved documents.

        This method overrides the default call behavior to allow for the inclusion of custom search arguments
        (`search_kwargs`) and manages the flow of retrieving documents and generating answers.

        Args:
            inputs (Dict[str, Any]): A dictionary containing input data, including the question and search_kwargs.
            run_manager (CallbackManagerForChainRun, optional): The callback manager to handle run callbacks.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and, if applicable, the source documents.
        '''
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        # Get search_kwargs conditions
        search_kwargs = inputs['search_kwargs']
        accepts_run_manager = (
            'run_manager' in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(question, search_kwargs,
                                  run_manager=_run_manager)
        else:
            docs = self._get_docs(question, search_kwargs,
                                  run_manager=_run_manager)
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        '''Asynchronously retrieve relevant documents for a given question.

        Args:
            question (str): The question for which to retrieve relevant documents.
            run_manager (AsyncCallbackManagerForChainRun): The callback manager to handle asynchronous run callbacks.
            search_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the search.

        Returns:
            List[Document]: A list of relevant documents retrieved based on the provided question.
        '''
        pass  # Implementation to be added

    async def _acall(
        self,
        search_kwargs: Dict[str, Any],
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        '''Asynchronously process input and generate an answer using the retrieved documents.

        This method handles the asynchronous retrieval of documents and generation of answers,
        allowing for non-blocking operations in a concurrent environment.

        Args:
            search_kwargs (Dict[str, Any]): Additional keyword arguments for the search.
            inputs (Dict[str, Any]): A dictionary containing input data, including the question and search_kwargs.
            run_manager (Optional[AsyncCallbackManagerForChainRun], optional): The callback manager to handle asynchronous run callbacks.

        Returns:
            Dict[str, Any]: A dictionary containing the answer and, if applicable, the source documents.

        Example:
            .. code-block:: python
                res = indexqa({'query': 'This is my query'})
                answer, docs = res['result'], res['source_documents']
        '''
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]
        search_kwargs = inputs.get("search_kwargs", None)
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._aget_docs).parameters
        )
        if accepts_run_manager:
            docs = await self._aget_docs(question, run_manager=_run_manager)
            docs = await self._aget_docs(
                question, run_manager=_run_manager, search_kwargs=search_kwargs
            )
        else:
            docs = await self._aget_docs(question)  # type: ignore[call-arg]
            # type: ignore[call-arg]
            docs = await self._aget_docs(question, search_kwargs=search_kwargs)
        answer = await self.combine_documents_chain.arun(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class VectorStoreRetrieverFilter(VectorStoreRetriever):
    '''Custom vector store retriever with filter functionality.

    This class provides a customized implementation of a vector store retriever, supporting different types of
    similarity searches and filtering based on custom search keyword arguments (`search_kwargs`). It supports
    both synchronous and asynchronous document retrieval.
    '''

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        '''Retrieve relevant documents from the vector store based on the query.

        This method supports various types of similarity searches (e.g., similarity, similarity with score threshold,
        maximal marginal relevance) and allows for customization using `search_kwargs`.

        Args:
            query (str): The query string for retrieving relevant documents.
            run_manager (CallbackManagerForRetrieverRun): The callback manager to handle retriever run callbacks.
            search_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the search.

        Returns:
            List[Document]: A list of relevant documents retrieved based on the provided query.
        '''
        merged_search_kwargs: dict = self.search_kwargs
        if search_kwargs is not None:
            if self.search_kwargs is not None:
                merged_search_kwargs = self.search_kwargs.copy()
                merged_search_kwargs.update(search_kwargs)
            else:
                merged_search_kwargs = search_kwargs
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, **merged_search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **merged_search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **merged_search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        '''Asynchronously retrieve relevant documents from the vector store based on the query.

        This method supports asynchronous document retrieval, allowing for non-blocking operations
        and various types of similarity searches.

        Args:
            query (str): The query string for retrieving relevant documents.
            run_manager (AsyncCallbackManagerForRetrieverRun): The callback manager to handle asynchronous retriever run callbacks.
            search_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the search.

        Returns:
            List[Document]: A list of relevant documents retrieved based on the provided query.
        '''
        merged_search_kwargs: dict = self.search_kwargs
        if search_kwargs is not None:
            if self.search_kwargs is not None:
                merged_search_kwargs = self.search_kwargs.copy()
                merged_search_kwargs.update(search_kwargs)
            else:
                merged_search_kwargs = search_kwargs
        if self.search_type == "similarity":
            docs = await self.vectorstore.asimilarity_search(
                query, **merged_search_kwargs
            )
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **merged_search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **merged_search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")