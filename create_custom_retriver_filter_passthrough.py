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
    '''Custom RetrievalQA Chain with filter functionality.'''

    def _get_docs(
        self,
        question: str,
        search_kwargs: Dict[str, Any] = None,
        *,
        run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        '''Overrided get docs.'''
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
        '''Overrided call method so we can provide search_kwargs.'''
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
        """Get documents to do question answering over."""

    async def _acall(
        self,
        search_kwargs: Dict[str, Any],
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.
        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.
        Example:
        .. code-block:: python
        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
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
    '''Custom vectorstore retriever with filter functionality.'''

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
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
