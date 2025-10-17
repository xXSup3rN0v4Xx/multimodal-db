"""
LlamaIndex Polars Query Engine Integration
Natural language queries on Polars DataFrames using LLMs for high-speed analytics.
"""
import polars as pl
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from llama_index.experimental.query_engine import PolarsQueryEngine
    from llama_index.core import PromptTemplate, Settings
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False


class PolarsNLQueryEngine:
    """
    Natural language query engine for Polars DataFrames.
    Uses LlamaIndex PolarsQueryEngine with Ollama for high-performance code generation.
    Optimized for streaming data and large datasets.
    """
    
    def __init__(self,
                 llm_model: str = "qwen2.5-coder:3b",
                 llm_base_url: str = "http://localhost:11434",
                 verbose: bool = False):
        """
        Initialize Polars Query Engine.
        
        Args:
            llm_model: Ollama model for query generation
            llm_base_url: Ollama API base URL
            verbose: Show generated polars code
        """
        self.available = LLAMAINDEX_AVAILABLE
        self.verbose = verbose
        
        if not self.available:
            return
        
        # Initialize Ollama LLM and set it in global Settings
        try:
            Settings.llm = Ollama(
                model=llm_model,
                base_url=llm_base_url,
                request_timeout=60.0
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Ollama LLM: {e}")
            self.available = False
    
    def load_dataframe_from_parquet(self, file_path: str) -> pl.DataFrame:
        """
        Load DataFrame from Parquet file.
        
        Args:
            file_path: Path to Parquet file
        
        Returns:
            Polars DataFrame
        """
        return pl.read_parquet(file_path)
    
    def create_query_engine(self,
                           df: pl.DataFrame,
                           instruction_str: Optional[str] = None,
                           synthesize_response: bool = True) -> Optional[Any]:
        """
        Create query engine for a Polars DataFrame.
        
        Args:
            df: Polars DataFrame to query
            instruction_str: Custom instructions for query generation
            synthesize_response: Use LLM to synthesize natural language response
        
        Returns:
            PolarsQueryEngine instance
        """
        if not self.available:
            return None
        
        try:
            # Create default instruction if not provided
            if instruction_str is None:
                instruction_str = """\
1. Convert the query to executable Python code using Polars.
2. The final line of code should be a Python expression that can be called with the `eval()` function.
3. The code should represent a solution to the query.
4. PRINT ONLY THE EXPRESSION.
5. Do not quote the expression.
"""
            
            # Create query engine - it uses Settings.llm automatically
            query_engine = PolarsQueryEngine(
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
             df: pl.DataFrame,
             query: str,
             synthesize_response: bool = True) -> Dict[str, Any]:
        """
        Execute natural language query on Polars DataFrame.
        
        Args:
            df: Polars DataFrame to query
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
            
            # Extract polars instruction if available
            polars_code = None
            if hasattr(response, 'metadata') and 'polars_instruction_str' in response.metadata:
                polars_code = response.metadata['polars_instruction_str']
            
            return {
                "success": True,
                "response": str(response),
                "polars_code": polars_code,
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
        Query data directly from Parquet file (zero-copy streaming).
        
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
                   df: pl.DataFrame,
                   queries: List[str],
                   synthesize_response: bool = True) -> List[Dict[str, Any]]:
        """
        Execute multiple queries on the same DataFrame efficiently.
        
        Args:
            df: Polars DataFrame to query
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
                
                polars_code = None
                if hasattr(response, 'metadata') and 'polars_instruction_str' in response.metadata:
                    polars_code = response.metadata['polars_instruction_str']
                
                results.append({
                    "success": True,
                    "query": query,
                    "response": str(response),
                    "polars_code": polars_code
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "query": query,
                    "error": str(e),
                    "response": None
                })
        
        return results
    
    def streaming_query(self,
                       df: pl.DataFrame,
                       query: str,
                       batch_size: int = 10000) -> Dict[str, Any]:
        """
        Execute query on large DataFrame using streaming/batching.
        Useful for YOLO detection data and other high-throughput streams.
        
        Args:
            df: Polars DataFrame to query
            query: Natural language query
            batch_size: Rows per batch for streaming
        
        Returns:
            Query results with streaming metadata
        """
        if not self.available:
            return {
                "success": False,
                "error": "LlamaIndex not available"
            }
        
        try:
            # For streaming queries, we'll process in batches
            total_rows = df.height
            num_batches = (total_rows + batch_size - 1) // batch_size
            
            # Create query engine for first batch to test
            first_batch = df.head(batch_size)
            result = self.query(first_batch, query, synthesize_response=True)
            
            return {
                **result,
                "streaming_info": {
                    "total_rows": total_rows,
                    "batch_size": batch_size,
                    "num_batches": num_batches,
                    "sample_batch_rows": batch_size
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_vision_detections(self,
                                  detections_file: str,
                                  analysis_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze YOLO vision detection data from Polars database.
        Optimized for high-speed streaming detection data.
        
        Args:
            detections_file: Path to detections Parquet file
            analysis_queries: Custom analysis queries
        
        Returns:
            Analysis results
        """
        if analysis_queries is None:
            analysis_queries = [
                "How many detections are there in total?",
                "What are the top 5 most detected object classes?",
                "What is the average confidence score?",
                "Show detections grouped by timestamp hour"
            ]
        
        try:
            df = self.load_dataframe_from_parquet(detections_file)
            results = self.batch_query(df, analysis_queries, synthesize_response=True)
            
            return {
                "file": detections_file,
                "total_detections": df.height,
                "columns": df.columns,
                "analyses": results
            }
        except Exception as e:
            return {
                "file": detections_file,
                "error": str(e)
            }
    
    def analyze_conversations(self,
                            conversations_file: str,
                            analysis_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze conversation data with Polars for high-speed processing.
        
        Args:
            conversations_file: Path to conversations Parquet file
            analysis_queries: Custom analysis queries
        
        Returns:
            Analysis results
        """
        if analysis_queries is None:
            analysis_queries = [
                "How many total conversations are there?",
                "What is the distribution of roles?",
                "What is the average message length by role?",
                "Show the most recent 5 conversations"
            ]
        
        try:
            df = self.load_dataframe_from_parquet(conversations_file)
            results = self.batch_query(df, analysis_queries, synthesize_response=True)
            
            return {
                "file": conversations_file,
                "total_records": df.height,
                "columns": df.columns,
                "analyses": results
            }
        except Exception as e:
            return {
                "file": conversations_file,
                "error": str(e)
            }
    
    def analyze_media(self,
                     media_file: str,
                     analysis_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze media metadata with Polars.
        
        Args:
            media_file: Path to media index Parquet file
            analysis_queries: Custom analysis queries
        
        Returns:
            Analysis results
        """
        if analysis_queries is None:
            analysis_queries = [
                "How many media files are there by type?",
                "What is the total file size by media type?",
                "What is the average file size per media type?",
                "Show the 10 largest media files"
            ]
        
        try:
            df = self.load_dataframe_from_parquet(media_file)
            results = self.batch_query(df, analysis_queries, synthesize_response=True)
            
            return {
                "file": media_file,
                "total_media": df.height,
                "columns": df.columns,
                "analyses": results
            }
        except Exception as e:
            return {
                "file": media_file,
                "error": str(e)
            }
    
    def aggregate_streaming_data(self,
                                df: pl.DataFrame,
                                group_by: List[str],
                                aggregations: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform aggregations on streaming data (e.g., YOLO detections).
        
        Args:
            df: Polars DataFrame
            group_by: Columns to group by
            aggregations: Dictionary of column -> aggregation function
        
        Returns:
            Aggregation results
        """
        try:
            # Build aggregation expressions
            agg_exprs = []
            for col, func in aggregations.items():
                if func == "count":
                    agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
                elif func == "mean":
                    agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif func == "sum":
                    agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif func == "max":
                    agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
                elif func == "min":
                    agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            
            # Perform aggregation
            result = df.group_by(group_by).agg(agg_exprs)
            
            return {
                "success": True,
                "result": result.to_dicts(),
                "shape": result.shape
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_prompts(self, df: pl.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Get the current prompts used by the query engine.
        
        Args:
            df: Polars DataFrame to create temporary engine
        
        Returns:
            Dictionary of prompt templates
        """
        if not self.available:
            return None
        
        try:
            engine = self.create_query_engine(df)
            if engine is None:
                return None
            
            prompts = engine.get_prompts()
            return {
                name: {"template": prompt.template if hasattr(prompt, 'template') else str(prompt)}
                for name, prompt in prompts.items()
            }
        except Exception as e:
            print(f"Error getting prompts: {e}")
            return None
    
    def custom_prompt_query(self,
                          df: pl.DataFrame,
                          query: str,
                          custom_prompt: str) -> Dict[str, Any]:
        """
        Query with custom prompt template.
        
        Args:
            df: Polars DataFrame
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
            engine.update_prompts({"polars_prompt": new_prompt})
            
            # Execute query
            response = engine.query(query)
            
            return {
                "success": True,
                "response": str(response),
                "polars_code": response.metadata.get('polars_instruction_str') if hasattr(response, 'metadata') else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }