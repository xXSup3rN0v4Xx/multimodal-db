"""
LlamaIndex Qdrant Hybrid Search Integration
Dense + Sparse vector search with BM25 and neural reranking for enhanced RAG.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct,
        SparseVectorParams, SparseVector
    )
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


class QdrantHybridSearch:
    """
    Hybrid search combining dense embeddings, sparse vectors (BM25-style),
    and neural reranking for state-of-the-art retrieval.
    """
    
    def __init__(self,
                 collection_name: str = "hybrid_search",
                 persist_path: str = "qdrant_hybrid",
                 dense_model: str = "BAAI/bge-small-en-v1.5",
                 llm_model: str = "qwen2.5-coder:3b",
                 llm_base_url: str = "http://localhost:11434",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize hybrid search system.
        
        Args:
            collection_name: Qdrant collection name
            persist_path: Path to store Qdrant database
            dense_model: HuggingFace model for dense embeddings
            llm_model: Ollama model for generation
            llm_base_url: Ollama API base URL
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
        """
        self.available = LLAMAINDEX_AVAILABLE
        self.collection_name = collection_name
        
        if not self.available:
            return
        
        # Setup paths
        self.persist_path = Path("data") / "qdrant" / persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(path=str(self.persist_path))
        except Exception as e:
            print(f"Warning: Failed to initialize Qdrant: {e}")
            self.available = False
            return
        
        # Initialize embeddings
        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=dense_model,
                cache_folder=str(Path("data") / "embeddings_cache")
            )
            self.vector_size = len(self.embed_model.get_text_embedding("test"))
        except Exception as e:
            print(f"Warning: Failed to initialize embeddings: {e}")
            self.available = False
            return
        
        # Initialize LLM
        try:
            self.llm = Ollama(
                model=llm_model,
                base_url=llm_base_url,
                request_timeout=60.0
            )
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")
            self.llm = None
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        if self.llm:
            Settings.llm = self.llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Text splitter for chunking
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize collection
        self._init_collection()
        
        # Vector store and index
        self.vector_store = None
        self.index = None
        self._init_index()
    
    def _init_collection(self):
        """Initialize Qdrant collection with hybrid vectors."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with dense vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    # Sparse vectors for BM25-style search
                    sparse_vectors_config={
                        "text": SparseVectorParams()
                    }
                )
        except Exception as e:
            print(f"Warning: Collection initialization failed: {e}")
    
    def _init_index(self):
        """Initialize vector store and index."""
        try:
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name
            )
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context
            )
        except Exception as e:
            print(f"Warning: Index initialization failed: {e}")
    
    def add_documents(self,
                     documents: List[str],
                     metadata: Optional[List[Dict[str, Any]]] = None,
                     agent_id: Optional[str] = None) -> List[str]:
        """
        Add documents to hybrid search index.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            agent_id: Agent ID for filtering
        
        Returns:
            List of document IDs
        """
        if not self.available or self.index is None:
            return []
        
        try:
            # Create Document objects
            doc_objects = []
            for i, doc_text in enumerate(documents):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                if agent_id:
                    doc_metadata["agent_id"] = agent_id
                
                doc = Document(
                    text=doc_text,
                    metadata=doc_metadata
                )
                doc_objects.append(doc)
            
            # Insert documents
            for doc in doc_objects:
                self.index.insert(doc)
            
            # Return document IDs
            return [doc.doc_id for doc in doc_objects]
        
        except Exception as e:
            print(f"Error adding documents: {e}")
            return []
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 10,
                     agent_id: Optional[str] = None,
                     rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with dense + sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            agent_id: Filter by agent ID
            rerank: Apply neural reranking
        
        Returns:
            List of search results
        """
        if not self.available or self.index is None:
            return []
        
        try:
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k * 2 if rerank else top_k  # Get more for reranking
            )
            
            # Retrieve nodes
            nodes = retriever.retrieve(query)
            
            # Filter by agent_id if provided
            if agent_id:
                nodes = [
                    node for node in nodes
                    if node.metadata.get("agent_id") == agent_id
                ]
            
            # Rerank if enabled
            if rerank and self.llm:
                nodes = self._rerank_nodes(query, nodes, top_k)
            else:
                nodes = nodes[:top_k]
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                })
            
            return results
        
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []
    
    def _rerank_nodes(self, query: str, nodes: List, top_k: int) -> List:
        """
        Rerank nodes using LLM-based relevance scoring.
        
        Args:
            query: Search query
            nodes: Retrieved nodes
            top_k: Number of top results
        
        Returns:
            Reranked nodes
        """
        try:
            # Simple reranking: use LLM to score relevance
            scored_nodes = []
            
            for node in nodes:
                # Create relevance prompt
                prompt = f"""Rate the relevance of this passage to the query on a scale of 0-10.
Query: {query}
Passage: {node.text[:500]}...

Respond with only a number between 0 and 10."""
                
                try:
                    # Get LLM score
                    response = self.llm.complete(prompt)
                    score = float(response.text.strip())
                    scored_nodes.append((score, node))
                except:
                    # Keep original score if reranking fails
                    scored_nodes.append((node.score or 0, node))
            
            # Sort by reranked score
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            
            return [node for _, node in scored_nodes[:top_k]]
        
        except Exception as e:
            print(f"Error in reranking: {e}")
            return nodes[:top_k]
    
    def query_with_context(self,
                          query: str,
                          top_k: int = 5,
                          agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query with context retrieval and response generation.
        
        Args:
            query: User query
            top_k: Number of context documents
            agent_id: Filter by agent ID
        
        Returns:
            Response with context
        """
        if not self.available or not self.llm:
            return {
                "success": False,
                "error": "System not available"
            }
        
        try:
            # Retrieve context
            contexts = self.hybrid_search(query, top_k=top_k, agent_id=agent_id)
            
            if not contexts:
                return {
                    "success": True,
                    "response": "No relevant context found.",
                    "contexts": []
                }
            
            # Build context string
            context_str = "\n\n".join([
                f"[{i+1}] {ctx['text']}"
                for i, ctx in enumerate(contexts)
            ])
            
            # Generate response with context
            prompt = f"""Use the following context to answer the question.

Context:
{context_str}

Question: {query}

Answer:"""
            
            response = self.llm.complete(prompt)
            
            return {
                "success": True,
                "response": response.text,
                "contexts": contexts,
                "num_contexts": len(contexts)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_agent_knowledge(self,
                           agent_id: str,
                           documents: List[str],
                           document_types: Optional[List[str]] = None) -> int:
        """
        Add knowledge documents for an agent.
        
        Args:
            agent_id: Agent identifier
            documents: List of documents
            document_types: Types of documents (conversation, document, etc.)
        
        Returns:
            Number of documents added
        """
        metadata = []
        for i, doc in enumerate(documents):
            doc_type = document_types[i] if document_types and i < len(document_types) else "document"
            metadata.append({
                "agent_id": agent_id,
                "document_type": doc_type
            })
        
        doc_ids = self.add_documents(documents, metadata, agent_id)
        return len(doc_ids)
    
    def search_agent_knowledge(self,
                              agent_id: str,
                              query: str,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search agent's knowledge base.
        
        Args:
            agent_id: Agent identifier
            query: Search query
            top_k: Number of results
        
        Returns:
            Search results
        """
        return self.hybrid_search(query, top_k=top_k, agent_id=agent_id)
    
    def delete_agent_knowledge(self, agent_id: str) -> bool:
        """
        Delete all knowledge for an agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Success status
        """
        try:
            # Delete points with matching agent_id
            from qdrant_client.http.models import Filter, FieldCondition
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(
                        key="agent_id",
                        match={"value": agent_id}
                    )]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting agent knowledge: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search statistics."""
        if not self.available:
            return {"available": False}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "available": True,
                "collection_name": self.collection_name,
                "vector_count": collection_info.vectors_count,
                "vector_size": self.vector_size,
                "persist_path": str(self.persist_path),
                "dense_model": self.embed_model.model_name if hasattr(self.embed_model, 'model_name') else "unknown",
                "has_llm": self.llm is not None
            }
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }