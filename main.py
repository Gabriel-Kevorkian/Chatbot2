import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress warnings and info messages (0=all,1=info,2=warning,3=error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from db import get_all_categories_and_brands
from langchain_core.messages import HumanMessage
import json
from langchain_core.messages import AIMessage
import time
import hashlib
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import threading

from tools import (
    list_brands_by_category,
    get_product_details,
    query_products,
    semantic_search_tool
)


# ===================== CACHING SYSTEM =====================

class ChatbotCache:
    def __init__(self, cache_dir="chatbot_cache", db_name="cache.db", expire_hours=24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.expire_hours = expire_hours
        self.lock = threading.Lock()
        self._init_db()

    def _adapt_datetime(self, dt):
        """Convert datetime to ISO format string for SQLite"""
        return dt.isoformat()

    def _convert_datetime(self, s):
        """Convert ISO format string back to datetime"""
        return datetime.fromisoformat(s.decode('utf-8') if isinstance(s, bytes) else s)

    def _init_db(self):
        """Initialize SQLite database for caching"""
        # Register datetime converters for Python 3.12+ compatibility
        sqlite3.register_adapter(datetime, self._adapt_datetime)
        sqlite3.register_converter("timestamp", self._convert_datetime)

        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tool_cache (
                    id INTEGER PRIMARY KEY,
                    cache_key TEXT UNIQUE,
                    tool_name TEXT,
                    result BLOB,
                    created_at TEXT,
                    expires_at TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS llm_cache (
                    id INTEGER PRIMARY KEY,
                    message_hash TEXT UNIQUE,
                    response BLOB,
                    created_at TEXT,
                    expires_at TEXT
                )
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tool_cache_key ON tool_cache(cache_key)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_llm_hash ON llm_cache(message_hash)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_tool_expires ON tool_cache(expires_at)
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_llm_expires ON llm_cache(expires_at)
            ''')

    def _generate_cache_key(self, tool_name: str, **kwargs) -> str:
        """Generate a unique cache key for tool calls"""
        # Sort and filter out None values for consistent caching
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        key_data = f"{tool_name}_{json.dumps(filtered_kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _generate_message_hash(self, messages: list) -> str:
        """Generate hash for message sequence"""
        # Only hash the last few messages to avoid cache misses due to long conversations
        recent_messages = messages[-2:] if len(messages) > 2 else messages
        message_content = []

        for msg in recent_messages:
            msg_type = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', '')

            # For tool calls, include tool name and args for better cache granularity
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    content += f"_TOOL:{tool_call['name']}:{json.dumps(tool_call['args'], sort_keys=True)}"

            message_content.append({"role": msg_type, "content": content})

        message_str = json.dumps(message_content, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()

    def get_tool_result(self, tool_name: str, **kwargs):
        """Get cached tool result"""
        cache_key = self._generate_cache_key(tool_name, **kwargs)
        current_time = datetime.now().isoformat()

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT result FROM tool_cache 
                    WHERE cache_key = ? AND expires_at > ?
                ''', (cache_key, current_time))

                row = cursor.fetchone()
                if row:
                    print(f"🚀 Cache HIT for {tool_name}")
                    return pickle.loads(row[0])

        print(f"💾 Cache MISS for {tool_name}")
        return None

    def store_tool_result(self, tool_name: str, result, **kwargs):
        """Store tool result in cache"""
        cache_key = self._generate_cache_key(tool_name, **kwargs)
        created_at = datetime.now().isoformat()
        expires_at = (datetime.max).isoformat()


        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO tool_cache 
                    (cache_key, tool_name, result, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (cache_key, tool_name, pickle.dumps(result), created_at, expires_at))

    def get_llm_response(self, messages: list):
        """Get cached LLM response"""
        message_hash = self._generate_message_hash(messages)
        current_time = datetime.now().isoformat()

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT response FROM llm_cache 
                    WHERE message_hash = ? AND expires_at > ?
                ''', (message_hash, current_time))

                row = cursor.fetchone()
                if row:
                    print("🚀 Cache HIT for LLM response")
                    return pickle.loads(row[0])

        print("💾 Cache MISS for LLM response")
        return None

    def store_llm_response(self, messages: list, response):
        """Store LLM response in cache"""
        message_hash = self._generate_message_hash(messages)
        created_at = datetime.now().isoformat()
        expires_at = (datetime.max).isoformat()

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO llm_cache 
                    (message_hash, response, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (message_hash, pickle.dumps(response), created_at, expires_at))

    def clear_expired(self):
        """Clear expired cache entries"""
        current_time = datetime.now().isoformat()

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                tool_deleted = conn.execute('DELETE FROM tool_cache WHERE expires_at < ?', (current_time,)).rowcount
                llm_deleted = conn.execute('DELETE FROM llm_cache WHERE expires_at < ?', (current_time,)).rowcount
                print(f"🧹 Cleared {tool_deleted} expired tool entries and {llm_deleted} expired LLM entries")

    def clear_all(self):
        """Clear all cache entries"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                tool_count = conn.execute('SELECT COUNT(*) FROM tool_cache').fetchone()[0]
                llm_count = conn.execute('SELECT COUNT(*) FROM llm_cache').fetchone()[0]
                conn.execute('DELETE FROM tool_cache')
                conn.execute('DELETE FROM llm_cache')
                print(f"🗑️ Cleared {tool_count} tool entries and {llm_count} LLM entries")

    def get_cache_stats(self):
        """Get cache statistics"""
        current_time = datetime.now().isoformat()

        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                tool_active = \
                conn.execute('SELECT COUNT(*) FROM tool_cache WHERE expires_at > ?', (current_time,)).fetchone()[0]
                tool_expired = \
                conn.execute('SELECT COUNT(*) FROM tool_cache WHERE expires_at <= ?', (current_time,)).fetchone()[0]
                llm_active = \
                conn.execute('SELECT COUNT(*) FROM llm_cache WHERE expires_at > ?', (current_time,)).fetchone()[0]
                llm_expired = \
                conn.execute('SELECT COUNT(*) FROM llm_cache WHERE expires_at <= ?', (current_time,)).fetchone()[0]

                # Get cache hit rates (would need to track this separately for accurate stats)
                total_size = conn.execute(
                    'SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()').fetchone()[0]

                return {
                    "tool_cache_active": tool_active,
                    "tool_cache_expired": tool_expired,
                    "llm_cache_active": llm_active,
                    "llm_cache_expired": llm_expired,
                    "cache_size_bytes": total_size
                }


# ===================== CACHED TOOLS =====================

class CachedToolWrapper:
    def __init__(self, tool, cache: ChatbotCache, tool_name: str):
        self.tool = tool
        self.cache = cache
        self.tool_name = tool_name

    def invoke(self, kwargs):
        # Try to get from cache first
        cached_result = self.cache.get_tool_result(self.tool_name, **kwargs)
        if cached_result is not None:
            return cached_result

        # If not in cache, execute tool and cache result
        result = self.tool.invoke(kwargs)
        self.cache.store_tool_result(self.tool_name, result, **kwargs)
        return result


# Initialize cache system
cache = ChatbotCache(expire_hours=24)  # Cache for 24 hours

# Initialize the LLM model
llm = init_chat_model("qwen2.5:7b-instruct-q4_K_M", model_provider="ollama")

# Wrap tools with caching
cached_list_brands = CachedToolWrapper(list_brands_by_category, cache, "list_brands_by_category")
cached_get_product_details = CachedToolWrapper(get_product_details, cache, "get_product_details")
cached_query_products = CachedToolWrapper(query_products, cache, "query_products")
cached_semantic_search = CachedToolWrapper(semantic_search_tool, cache, "semantic_search_tool")

# Bind all tools to the LLM
llm = llm.bind_tools([
    list_brands_by_category,
    get_product_details,
    query_products,
    semantic_search_tool
])


# Define the state schema
class ChatState(TypedDict):
    messages: list


# Outside llm_node, after fetching categories and brands:
categories, brands = get_all_categories_and_brands()

system_message_content = f"""
You are an intelligent and strictly grounded e-commerce assistant.

You help users find and explore products **only from our internal database**, using the following tools:
You MUST NOT answer any product-related queries without calling the `query_products` tool.
Never generate product names, descriptions, or prices yourself. Always rely on the database results.

---

🧰 Tools Available:

1. query_products(category, brand, gender, size, min_price, max_price, in_stock=True, limit=3)
    - Use to search products with filters (category, brand, gender, size, price).
    - Always use this tool when the user asks about products.
    - Never make up product names or prices.

2. get_product_details(product_name, include_images=False)
    - Use this to get more info on a specific known product.
    - You must know the exact product name from the database (don't guess).

3. list_brands_by_category(category_name)
    - Use this to help the user choose a brand in a given category.
    - Only if they ask for brand options or you're unsure what brand to filter by.
4. semantic_search_tool(query, top_k=3)
    - Use this when the user provides unstructured, vague, or natural language queries that don't specify exact filters (e.g., "something elegant", "a cool outfit", "shoes that go with jeans").
    - Never use this when structured filters like category and brand are available.
    - Return only the top 3 most relevant product results based on meaning.
---

📌 Decision Rules:
User can ask for more products in many ways, such as:
- "more"
- "show me other options"
- "anything else"
- "more products"
- "different styles"
- "other brands"

For all these requests, you must:
- Call the `query_products` tool with the same filters but with an increased offset (to fetch the next batch of products).
- Avoid repeating products already shown.

**IMPORTANT FALLBACK BEHAVIOR:**
- If you just said "I couldn't find any products matching those specific criteria. Let me try a semantic search to find similar products that might interest you.", you MUST immediately call the `semantic_search_tool` with a natural language query based on the user's original request.
- Reformulate the user's original product request into a descriptive query for semantic search.
- For example: if user asked for "red Nike shoes" and structured search failed, use semantic_search_tool with query like "red athletic shoes Nike style" or "red sneakers similar to Nike".


- You must **always use tools** for product information. Never guess or generate product data.
- If the user's message is ambiguous (e.g., "show me shirts"), use `query_products` with just the category.
- If the user's message is ambiguous or vague (e.g., "show me something nice"), call `semantic_search_tool`.
- If the user asks for more info about a known product, use `get_product_details` with the exact name.
- If you're unsure which brand to search for in a category, call `list_brands_by_category` first.
- Always provide **3 products at most** unless the user asks for more.
- If the user ask for shoes or shirts always ask if for women or men.
- If the user ask for "cars" search in the vehicle category.
---

📊 Available Categories: {', '.join(categories)}
🏷️ Available Brands: {', '.join(brands)}

IMPORTANT: You must NOT generate or infer any product information on your own.  
You can only repeat and explain the data that comes directly from the tools.  
If the tools return no data, respond only with:  
"I'm sorry, I couldn't find any matching products in our database."
If you couldn't found any products in the database using the query_products tool, use directly the semantic_searchtool.
Do NOT make up product names, descriptions, or prices under any circumstances.

Your goal is to use tools correctly and only reply based on what the tools return.
"""


def llm_node(state):
    system_message = {
        'role': 'system',
        'content': system_message_content
    }
    messages_with_system = [system_message] + state['messages']

    # Try to get cached LLM response
    cached_response = cache.get_llm_response(messages_with_system)
    if cached_response is not None:
        return {'messages': state['messages'] + [cached_response]}

    # If not cached, get new response and cache it
    response = llm.invoke(messages_with_system)
    cache.store_llm_response(messages_with_system, response)
    return {'messages': state['messages'] + [response]}


# Router: decides whether to call tools node or end
def router(state):
    last_message = state['messages'][-1]
    print("[Router Debug] Last message:", last_message)
    tool_calls = getattr(last_message, 'tool_calls', None)
    print("[Router Debug] Tool calls:", tool_calls)
    if tool_calls:
        return 'tools'
    else:
        return 'end'


# ToolNode setup with cached tools
tool_node = ToolNode([
    list_brands_by_category,
    get_product_details,
    query_products,
    semantic_search_tool
])


def retry_tool_call(tool_node, state, max_retries=3, delay=1):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Retry] Attempt {attempt}...")
            result = tool_node.invoke(state)
            return result
        except Exception as e:
            last_exception = e
            print(f"⚠️ Tool call failed (Attempt {attempt}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Tool call failed after {max_retries} attempts") from last_exception


def tools_node(state):
    print("[Tools Node] Invoked")

    # Get the last tool call to determine which cached tool to use
    last_tool_call = next(
        (
            msg.tool_calls[0]
            for msg in reversed(state["messages"])
            if hasattr(msg, "tool_calls") and msg.tool_calls
        ),
        None
    )

    if last_tool_call:
        tool_name = last_tool_call["name"]
        tool_args = last_tool_call["args"]

        # Use cached version of the tool
        try:
            if tool_name == "list_brands_by_category":
                result = cached_list_brands.invoke(tool_args)
            elif tool_name == "get_product_details":
                result = cached_get_product_details.invoke(tool_args)
            elif tool_name == "query_products":
                result = cached_query_products.invoke(tool_args)
            elif tool_name == "semantic_search_tool":
                result = cached_semantic_search.invoke(tool_args)
            else:
                # Fallback to original tool execution
                result = retry_tool_call(tool_node, state)
                return result

            # Create proper message format for the result
            from langchain_core.messages import ToolMessage
            tool_message = ToolMessage(
                content=json.dumps(result) if not isinstance(result, str) else result,
                tool_call_id=last_tool_call["id"]
            )

            new_messages = [tool_message]

        except Exception as e:
            print("❌ Cached tool call failed:", e)
            # Fallback to original tool execution
            try:
                result = retry_tool_call(tool_node, state)
                return result
            except Exception as e2:
                print("❌ Original tool call also failed:", e2)
                return {
                    "messages": state["messages"] + [
                        AIMessage(
                            content="Something went wrong while fetching the product information. Please try again.")
                    ]
                }
    else:
        # No tool call found, use original logic
        try:
            result = retry_tool_call(tool_node, state)
        except Exception as e:
            print("❌ Tool call ultimately failed after retries:", e)
            return {
                "messages": state["messages"] + [
                    AIMessage(content="Something went wrong while fetching the product information. Please try again.")
                ]
            }
        new_messages = result["messages"]

    # Handle fallback logic - but now trigger LLM to call semantic_search instead of doing it directly
    last_tool_response = new_messages[-1] if new_messages else None
    should_fallback = False
    tool_response_data = None

    try:
        tool_response_data = getattr(last_tool_response, "content", None)
        print("🔧 Raw tool response:", tool_response_data)

        # Only try to parse as JSON for tools that return JSON (not semantic_search)
        if last_tool_call and last_tool_call["name"] != "semantic_search_tool":
            if isinstance(tool_response_data, str):
                try:
                    tool_response_data = json.loads(tool_response_data)
                except json.JSONDecodeError as e:
                    print("⚠️ JSON decode error:", e)
                    tool_response_data = {}

            print("🔧 Parsed tool response:", tool_response_data)

            if last_tool_call["name"] == "query_products":
                status = tool_response_data.get("status")
                if status == "empty":
                    should_fallback = True
                elif status == "end_of_list":
                    print("That's all the products we have in this category.")
                    additional_message = AIMessage(content="That's all the products we have in this category.")
                    new_messages.append(additional_message)
        else:
            # For semantic_search_tool, the response is already formatted, no JSON parsing needed
            print("🔧 Semantic search response (no parsing needed):", tool_response_data)

    except Exception as e:
        print("⚠️ Failed to parse tool response:", e)

    # Modified fallback logic - add a special message to trigger semantic search
    if should_fallback:
        print("[Fallback Triggered] Prompting LLM to use semantic_search...")

        # Add a special system-like message that instructs the LLM to use semantic_search
        fallback_instruction = AIMessage(
            content="I couldn't find any products matching those specific criteria. Let me try a semantic search to find similar products that might interest you."
        )

        # Add this instruction message to trigger the LLM to make a semantic search call
        new_messages.append(fallback_instruction)

    return {"messages": state["messages"] + new_messages}


# Build the LangGraph state machine
builder = StateGraph(ChatState)
builder.add_node('llm', llm_node)
builder.add_node('tools', tools_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {
    'tools': 'tools',
    'end': END,
})

graph = builder.compile()

# Main interactive loop with cache management
if __name__ == "__main__":
    state = {'messages': []}
    print("Welcome to your Enhanced E-Commerce Assistant with Caching! 🚀")
    print("Type your question, 'cache_stats' to see cache info, 'clear_cache' to clear cache, or 'quit' to exit.\n")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'cache_stats':
            stats = cache.get_cache_stats()
            print(f"📊 Cache Stats:")
            print(f"   Active Tool Entries: {stats['tool_cache_active']}")
            print(f"   Expired Tool Entries: {stats['tool_cache_expired']}")
            print(f"   Active LLM Entries: {stats['llm_cache_active']}")
            print(f"   Expired LLM Entries: {stats['llm_cache_expired']}")
            print(f"   Cache Size: {stats['cache_size_bytes'] / 1024:.1f} KB")
            continue
        elif user_input.lower() == 'clear_cache':
            cache.clear_all()
            continue
        elif user_input.lower() == 'clear_expired':
            cache.clear_expired()
            continue

        # Add user message to the conversation state
        state['messages'].append(HumanMessage(content=user_input))

        # Run the LangGraph pipeline (LLM + tools) with caching
        start_time = time.time()
        state = graph.invoke(state)
        end_time = time.time()

        # Print assistant's response with timing
        print(f"{state['messages'][-1].content}")
        print(f"⏱️ Response time: {end_time - start_time:.2f}s\n")