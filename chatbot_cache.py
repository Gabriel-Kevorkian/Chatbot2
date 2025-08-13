import hashlib
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import threading
import json
class ChatbotCache:
    def __init__(self, cache_dir="chatbot_cache", db_name="cache.db", expire_minutes=5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.expire_minutes = expire_minutes
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
        expires_at = (datetime.now() + timedelta(minutes=self.expire_minutes)).isoformat()


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
        expires_at = (datetime.now() + timedelta(minutes=self.expire_minutes)).isoformat()

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