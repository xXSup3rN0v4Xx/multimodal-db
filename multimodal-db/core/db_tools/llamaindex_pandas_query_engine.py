"""
LlamaIndex Pandas Query Engine Integration
Natural language queries on Pandas DataFrames using LLMs.
"""
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.core import PromptTemplate, Settings
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

class PandasNLQueryEngine:
    """
    Natural language query engine for Pandas DataFrames.
    Uses LlamaIndex PandasQueryEngine with Ollama for code generation.
    """
    
    def __init__(self, 
                 llm_model: str = "qwen2.5-coder:3b",
                 llm_base_url: str = "http://localhost:11434",
                 verbose: bool = False):
        """
        Initialize Pandas Query Engine.
        
        Args:
            llm_model: Ollama model for query generation
            llm_base_url: Ollama API base URL
            verbose: Show generated pandas code
        """
        self.available = LLAMAINDEX_AVAILABLE
        self.verbose = verbose
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        
        if not self.available:
            return
        
        # Configure global Settings for LlamaIndex
        try:
            llm = Ollama(
                model=llm_model,
                base_url=llm_base_url,
                request_timeout=60.0
            )
            Settings.llm = llm
        except Exception as e:
            print(f"Warning: Failed to initialize Ollama LLM: {e}")
            self.available = False
    
    def load_dataframe_from_parquet(self, file_path: str) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file.
        
        Args:
            file_path: Path to Parquet file
        
        Returns:
            Pandas DataFrame
        """
        return pd.read_parquet(file_path)
    
    def create_query_engine(self, 
                           df: pd.DataFrame,
                           instruction_str: Optional[str] = None,
                           synthesize_response: bool = True) -> Optional[Any]:
        """
        Create query engine for a DataFrame.
        
        Args:
            df: Pandas DataFrame to query
            instruction_str: Custom instructions for query generation
            synthesize_response: Use LLM to synthesize natural language response
        
        Returns:
            PandasQueryEngine instance
        """
        if not self.available:
            return None
        
        try:
            # Create default instruction if not provided
            if instruction_str is None:
                instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""
            
            # Create query engine - it uses Settings.llm automatically
            query_engine = PandasQueryEngine(
                df=df,
                verbose=self.verbose,
                synthesize_response=synthesize_response,
                instruction_str=instruction_str
            )
            
            return query_engine
        except Exception as e:
            print(f"Error creating query engine: {e}")
            return None
    
    def query(self, 
             df: pd.DataFrame,
             query: str,
             synthesize_response: bool = True) -> Dict[str, Any]:
        """
        Execute natural language query on DataFrame.
        
        Args:
            df: Pandas DataFrame to query
            query: Natural language query
            synthesize_response: Use LLM for response synthesis
        
        Returns:
            Dictionary with response and metadata
        """
        if not self.available:
            return {
                "success": False,
                "error": "LlamaIndex not available",
                "response": None
            }
        
        try:
            # Create query engine
            engine = self.create_query_engine(df, synthesize_response=synthesize_response)
            if engine is None:
                return {
                    "success": False,
                    "error": "Failed to create query engine",
                    "response": None
                }
            
            # Execute query
            response = engine.query(query)
            
            # Extract pandas instruction if available
            pandas_code = None
            if hasattr(response, 'metadata') and 'pandas_instruction_str' in response.metadata:
                pandas_code = response.metadata['pandas_instruction_str']
            
            return {
                "success": True,
                "response": str(response),
                "pandas_code": pandas_code,
                "metadata": response.metadata if hasattr(response, 'metadata') else {}
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def query_from_parquet(self,
                          file_path: str,
                          query: str,
                          synthesize_response: bool = True) -> Dict[str, Any]:
        """
        Query data directly from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            query: Natural language query
            synthesize_response: Use LLM for response synthesis
        
        Returns:
            Query results dictionary
        """
        try:
            df = self.load_dataframe_from_parquet(file_path)
            return self.query(df, query, synthesize_response)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load Parquet file: {e}",
                "response": None
            }
    
    def batch_query(self,
                   df: pd.DataFrame,
                   queries: List[str],
                   synthesize_response: bool = True) -> List[Dict[str, Any]]:
        """
        Execute multiple queries on the same DataFrame.
        
        Args:
            df: Pandas DataFrame to query
            queries: List of natural language queries
            synthesize_response: Use LLM for response synthesis
        
        Returns:
            List of query result dictionaries
        """
        results = []
        
        # Create query engine once for efficiency
        engine = self.create_query_engine(df, synthesize_response=synthesize_response)
        if engine is None:
            return [{
                "success": False,
                "error": "Failed to create query engine",
                "query": q
            } for q in queries]
        
        for query in queries:
            try:
                response = engine.query(query)
                
                pandas_code = None
                if hasattr(response, 'metadata') and 'pandas_instruction_str' in response.metadata:
                    pandas_code = response.metadata['pandas_instruction_str']
                
                results.append({
                    "success": True,
                    "query": query,
                    "response": str(response),
                    "pandas_code": pandas_code
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "query": query,
                    "error": str(e),
                    "response": None
                })
        
        return results
    
    def analyze_conversations(self,
                            conversations_file: str,
                            analysis_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze conversation data from exported Parquet file.
        
        Args:
            conversations_file: Path to conversations Parquet file
            analysis_queries: Custom analysis queries (uses defaults if None)
        
        Returns:
            Analysis results
        """
        if analysis_queries is None:
            analysis_queries = [
                "How many total conversations are there?",
                "What is the distribution of roles (user vs assistant)?",
                "What is the average length of messages?",
                "Show the most recent 5 conversations by timestamp"
            ]
        
        try:
            df = self.load_dataframe_from_parquet(conversations_file)
            results = self.batch_query(df, analysis_queries, synthesize_response=True)
            
            return {
                "file": conversations_file,
                "total_records": len(df),
                "columns": list(df.columns),
                "analyses": results
            }
        except Exception as e:
            return {
                "file": conversations_file,
                "error": str(e)
            }
    
    def analyze_agents(self,
                      agents_file: str,
                      analysis_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze agent data from exported Parquet file.
        
        Args:
            agents_file: Path to agents Parquet file
            analysis_queries: Custom analysis queries
        
        Returns:
            Analysis results
        """
        if analysis_queries is None:
            analysis_queries = [
                "How many agents are there in total?",
                "What are the unique agent names?",
                "Show the distribution of agents by tags",
                "What is the oldest agent by created_at?"
            ]
        
        try:
            df = self.load_dataframe_from_parquet(agents_file)
            results = self.batch_query(df, analysis_queries, synthesize_response=True)
            
            return {
                "file": agents_file,
                "total_agents": len(df),
                "columns": list(df.columns),
                "analyses": results
            }
        except Exception as e:
            return {
                "file": agents_file,
                "error": str(e)
            }
    
    def custom_prompt_query(self,
                          df: pd.DataFrame,
                          query: str,
                          custom_prompt: str) -> Dict[str, Any]:
        """
        Query with custom prompt template.
        
        Args:
            df: Pandas DataFrame
            query: Natural language query
            custom_prompt: Custom prompt template
        
        Returns:
            Query results
        """
        if not self.available:
            return {
                "success": False,
                "error": "LlamaIndex not available"
            }
        
        try:
            # Create query engine with custom prompt
            engine = self.create_query_engine(df)
            if engine is None:
                return {
                    "success": False,
                    "error": "Failed to create query engine"
                }
            
            # Update prompt
            new_prompt = PromptTemplate(custom_prompt)
            engine.update_prompts({"pandas_prompt": new_prompt})
            
            # Execute query
            response = engine.query(query)
            
            return {
                "success": True,
                "response": str(response),
                "pandas_code": response.metadata.get('pandas_instruction_str') if hasattr(response, 'metadata') else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }