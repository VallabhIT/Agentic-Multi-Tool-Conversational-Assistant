# Agentic-Multi-Tool-Conversational-Assistant
Agentic conversational AI system capable of dynamically using multiple tools including private knowledge bases, web search, SQL queries, and Python execution.

* Embeddings Model: SentenceTransformerEmbeddings (all-MiniLM-L6-v2 - 384 dimensions)
* Vector Database: Chroma (open-source, local persistence with persist_directory)
* Text Splitting: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)
* LLM Model: ChatOllama (llama3.2:3b) - local/open-source alternative




Architecture Diagram-

```
User Input
      ↓
[ConversationalRetrievalChain OR AgentExecutor]
      ↓
Determine Tool(s) Needed
├─→ [NotionKnowledgeBase] (Chroma + SentenceTransformer)
├─→ [WebSearch] (DuckDuckGo)
├─→ [SQLExecutor] (Database)
└─→ [PythonREPL] (Code Execution)
      ↓
Retrieve/Execute Tool
      ↓
[ChatOllama] - Generate Response
      ↓
[ConversationBufferMemory] - Store History
      ↓
Response to User
```
