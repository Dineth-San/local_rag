"""Single source of truth for models, hosts, and tunable knobs."""

# Services
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

# Models
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text output size
LLM_MODEL = "llama3.2"
SPARSE_MODEL = "Qdrant/bm25"

# Named vectors in the Qdrant collection
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Semantic chunking
BREAKPOINT_TYPE = "percentile"  # or "standard_deviation", "interquartile", "gradient"
BREAKPOINT_AMOUNT = 90          # percentile cutoff; lower = more splits
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP = 100             # only used when force-splitting oversized chunks

# Summary generation
SUMMARY_SAMPLE_CHARS = 6000     # how much document text the LLM sees
LLM_NUM_CTX = 8192

# Retrieval / re-ranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVE_K = 40          # candidates pulled from Qdrant
RERANK_TOP_N = 5         # chunks sent to the LLM
RERANK_MIN_SCORE = None  # tune in step 5 (see below)

MAX_TOOL_CALLS = 3
HISTORY_TURNS = 6
AGENT_TEMPERATURE = 0
MEMORY_DB_PATH = "chat_memory.sqlite"