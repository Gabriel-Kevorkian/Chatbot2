import sqlite3
import threading
import uuid
from datetime import datetime,timedelta
import json
from typing import List, Optional, Dict


class ConversationManager:
    def __init__(self, db_path="chatbot_cache/conversations.db"):
        self.db_path = db_path
        self._init_conversation_db()

    def _init_conversation_db(self):
        """Initialize conversation database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    total_messages INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    timestamp TEXT NOT NULL,
                    response_time REAL,
                    cached BOOLEAN DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                );

                CREATE INDEX IF NOT EXISTS idx_conv_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_conv_status ON conversations(status);
            ''')

    def start_conversation(self, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Start a new conversation and return conversation ID"""
        conversation_id = str(uuid.uuid4())
        started_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversations (conversation_id, user_id, started_at, metadata)
                VALUES (?, ?, ?, ?)
            ''', (conversation_id, user_id, started_at, json.dumps(metadata) if metadata else None))

        print(f"📝 Started new conversation: {conversation_id[:8]}...")
        return conversation_id

    def save_message(self, conversation_id: str, message_type: str, content: str,
                     tool_calls: Optional[List] = None, response_time: Optional[float] = None,
                     cached: bool = False):
        """Save a message to the conversation"""
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO messages (conversation_id, message_type, content, tool_calls, 
                                    timestamp, response_time, cached)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, message_type, content,
                  json.dumps(tool_calls) if tool_calls else None,
                  timestamp, response_time, cached))

    def end_conversation(self, conversation_id: str):
        """Mark a conversation as ended and update statistics"""
        ended_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get message count
            message_count = conn.execute('''
                SELECT COUNT(*) FROM messages WHERE conversation_id = ?
            ''', (conversation_id,)).fetchone()[0]

            # Update conversation
            conn.execute('''
                UPDATE conversations 
                SET ended_at = ?, total_messages = ?, status = 'completed'
                WHERE conversation_id = ?
            ''', (ended_at, message_count, conversation_id))

        print(f"✅ Ended conversation: {conversation_id[:8]}... ({message_count} messages)")

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a specific conversation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT message_type, content, tool_calls, timestamp, response_time, cached
                FROM messages 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (conversation_id,))

            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'type': row[0],
                    'content': row[1],
                    'tool_calls': json.loads(row[2]) if row[2] else None,
                    'timestamp': row[3],
                    'response_time': row[4],
                    'cached': bool(row[5])
                })
            return messages

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations with basic info"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT conversation_id, user_id, started_at, ended_at, total_messages, status
                FROM conversations 
                ORDER BY started_at DESC 
                LIMIT ?
            ''', (limit,))

            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'conversation_id': row[0],
                    'user_id': row[1],
                    'started_at': row[2],
                    'ended_at': row[3],
                    'total_messages': row[4],
                    'status': row[5]
                })
            return conversations

    def search_conversations(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search conversations by content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT DISTINCT c.conversation_id, c.started_at, c.total_messages, 
                       c.status, m.content
                FROM conversations c
                JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE m.content LIKE ?
                ORDER BY c.started_at DESC
                LIMIT ?
            ''', (f'%{search_term}%', limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'conversation_id': row[0],
                    'started_at': row[1],
                    'total_messages': row[2],
                    'status': row[3],
                    'matching_content': row[4][:100] + '...' if len(row[4]) > 100 else row[4]
                })
            return results

    def get_conversation_stats(self) -> Dict:
        """Get overall conversation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total conversations
            total_conversations = conn.execute('SELECT COUNT(*) FROM conversations').fetchone()[0]

            # Active vs completed
            active_conversations = conn.execute(
                'SELECT COUNT(*) FROM conversations WHERE status = "active"'
            ).fetchone()[0]

            # Total messages
            total_messages = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]

            # Average messages per conversation
            avg_messages = conn.execute('''
                SELECT AVG(total_messages) FROM conversations WHERE total_messages > 0
            ''').fetchone()[0] or 0

            # Most active conversation
            most_active = conn.execute('''
                SELECT conversation_id, total_messages 
                FROM conversations 
                ORDER BY total_messages DESC 
                LIMIT 1
            ''').fetchone()

            # Cache hit rate for conversations
            cached_messages = conn.execute('SELECT COUNT(*) FROM messages WHERE cached = 1').fetchone()[0]
            cache_hit_rate = (cached_messages / total_messages * 100) if total_messages > 0 else 0

            return {
                'total_conversations': total_conversations,
                'active_conversations': active_conversations,
                'completed_conversations': total_conversations - active_conversations,
                'total_messages': total_messages,
                'avg_messages_per_conversation': round(avg_messages, 1),
                'most_active_conversation': most_active[0][:8] + '...' if most_active else None,
                'most_active_message_count': most_active[1] if most_active else 0,
                'cache_hit_rate': round(cache_hit_rate, 1)
            }

    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up conversations older than specified days"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get conversations to delete
            old_conversations = conn.execute('''
                SELECT conversation_id FROM conversations 
                WHERE started_at < ? AND status = 'completed'
            ''', (cutoff_date,)).fetchall()

            if old_conversations:
                old_conv_ids = [row[0] for row in old_conversations]

                # Delete messages first
                for conv_id in old_conv_ids:
                    conn.execute('DELETE FROM messages WHERE conversation_id = ?', (conv_id,))

                # Delete conversations
                conn.execute('''
                    DELETE FROM conversations 
                    WHERE started_at < ? AND status = 'completed'
                ''', (cutoff_date,))

                print(f"🗑️ Cleaned up {len(old_conv_ids)} conversations older than {days_old} days")


class ConversationStateManager:
    def __init__(self, max_conversations=100, max_messages_per_conv=50):
        """
        Manages multiple conversation states in memory

        Args:
            max_conversations: Maximum number of active conversations to keep in memory
            max_messages_per_conv: Maximum messages per conversation to keep in memory
        """
        self.conversations = {}  # {conversation_id: {'messages': [], 'last_active': datetime}}
        self.max_conversations = max_conversations
        self.max_messages_per_conv = max_messages_per_conv
        self.lock = threading.RLock()


    def get_conversation_state(self, conversation_id: str) -> Dict:
        """Get or create conversation state"""
        with self.lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = {
                    'messages': [],
                    'last_active': datetime.now(),
                    'context': {}
                }
                print(f"📝 Created new conversation state: {conversation_id[:8]}...")

            # Update last active time
            self.conversations[conversation_id]['last_active'] = datetime.now()

            # Clean up old conversations if needed
            self._cleanup_old_conversations()

            return self.conversations[conversation_id]

    def add_message_to_conversation(self, conversation_id: str, message):
        """Add a message to specific conversation"""
        with self.lock:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = {
                    'messages': [],
                    'last_active': datetime.now(),
                    'context': {}
                }

            conv_state = self.conversations[conversation_id]
            conv_state['messages'].append(message)
            conv_state['last_active'] = datetime.now()
            # Trim messages if conversation gets too long
            if len(conv_state['messages']) > self.max_messages_per_conv:
                # Keep system message if it exists, trim others
                messages = conv_state['messages']
                if len(messages) > 0 and getattr(messages[0], 'type', '') == 'system':
                    conv_state['messages'] = [messages[0]] + messages[-(self.max_messages_per_conv - 1):]
                else:
                    conv_state['messages'] = messages[-self.max_messages_per_conv:]
                print(f"✂️ Trimmed conversation {conversation_id[:8]}... to {len(conv_state['messages'])} messages")

    def get_conversation_messages(self, conversation_id: str) -> List:
        """Get messages for specific conversation"""
        with self.lock:  # Add this
            conv_state = self.get_conversation_state(conversation_id)
            print(f"Conversation:{conv_state['messages'].copy()}")
            return conv_state['messages'].copy()

    def update_conversation_context(self, conversation_id: str, context: Dict):
        """Update conversation context"""
        with self.lock:
            conv_state = self.get_conversation_state(conversation_id)
            conv_state['context'].update(context)

    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context"""
        conv_state = self.get_conversation_state(conversation_id)
        return conv_state['context'].copy()

    def _cleanup_old_conversations(self):
        """Clean up least recently used conversations if we have too many"""
        if len(self.conversations) <= self.max_conversations:
            return

        # Sort by last_active time and remove oldest
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1]['last_active']
        )

        conversations_to_remove = len(self.conversations) - self.max_conversations
        for i in range(conversations_to_remove):
            conv_id = sorted_convs[i][0]
            del self.conversations[conv_id]
            print(f"🗑️ Cleaned up inactive conversation: {conv_id[:8]}...")

    def remove_conversation(self, conversation_id: str):
        """Remove conversation from memory"""
        with self.lock:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                print(f"🗑️ Removed conversation from memory: {conversation_id[:8]}...")

    def get_active_conversations(self) -> List[Dict]:
        """Get list of active conversations"""
        with self.lock:
            active_convs = []
            for conv_id, conv_data in self.conversations.items():
                active_convs.append({
                    'conversation_id': conv_id,
                    'message_count': len(conv_data['messages']),
                    'last_active': conv_data['last_active'].isoformat(),
                    'has_context': bool(conv_data['context'])
                })

            # Sort by last active (most recent first)
            active_convs.sort(key=lambda x: x['last_active'], reverse=True)
            return active_convs
