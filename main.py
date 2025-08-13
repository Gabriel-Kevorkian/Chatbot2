import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress warnings and info messages (0=all,1=info,2=warning,3=error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
from typing import TypedDict,Optional
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from db import get_all_categories_and_brands
from langchain_core.messages import HumanMessage, ToolMessage
import json
from langchain_core.messages import AIMessage
import time
from datetime import datetime
from conversation_manager import ConversationManager,ConversationStateManager
from tools import (
    list_brands_by_category,
    get_product_details,
    query_products,
    semantic_search_tool
)


from chatbot_cache import ChatbotCache,CachedToolWrapper


conversation_manager = ConversationManager()
conv_state_manager = ConversationStateManager()



# Initialize cache system
cache = ChatbotCache(expire_minutes=5)

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
    conversation_id: Optional[str]

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
If you couldn't found any products in the database using the query_products tool, use directly the semantic_search_tool.
Do NOT make up product names, descriptions, or prices under any circumstances.

Your goal is to use tools correctly and only reply based on what the tools return.
"""


def llm_node(state):
    conversation_id = state.get('conversation_id')
    if not conversation_id:
        raise ValueError("conversation_id is required in state")

    # Get conversation-specific messages
    conversation_messages = conv_state_manager.get_conversation_messages(conversation_id)
    system_message = {'role': 'system', 'content': system_message_content}
    messages_with_system = [system_message] + conversation_messages

    # Try to get cached LLM response
    start_time = time.time()
    cached_response = cache.get_llm_response(messages_with_system)
    response_time = time.time() - start_time

    if cached_response is not None:
        # Add response to conversation state
        conv_state_manager.add_message_to_conversation(conversation_id, cached_response)

        # Save to database
        conversation_manager.save_message(
            conversation_id,
            'assistant',
            cached_response.content,
            getattr(cached_response, 'tool_calls', None),
            response_time,
            cached=True
        )

        # Return updated state with conversation-specific messages
        return {
            'messages': conv_state_manager.get_conversation_messages(conversation_id),
            'conversation_id': conversation_id
        }

    # If not cached, get new response
    response = llm.invoke(messages_with_system)
    response_time = time.time() - start_time
    cache.store_llm_response(messages_with_system, response)

    # Add response to conversation state
    conv_state_manager.add_message_to_conversation(conversation_id, response)

    # Save to database
    conversation_manager.save_message(
        conversation_id,
        'assistant',
        response.content,
        getattr(response, 'tool_calls', None),
        response_time,
        cached=False
    )

    return {
        'messages': conv_state_manager.get_conversation_messages(conversation_id),
        'conversation_id': conversation_id
    }


def multi_conv_tools_node(state):
    """Tools node that handles multiple conversations correctly"""
    conversation_id = state.get('conversation_id')
    if not conversation_id:
        raise ValueError("conversation_id is required in state")

    print(f"[Tools Node] Processing for conversation: {conversation_id[:8]}...")

    # Get conversation-specific messages
    conversation_messages = conv_state_manager.get_conversation_messages(conversation_id)

    # Find the last tool call in this conversation
    last_tool_call = None
    for msg in reversed(conversation_messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            last_tool_call = msg.tool_calls[0]
            break

    if not last_tool_call:
        print("❌ No tool call found")
        error_message = AIMessage(content="I couldn't process your request. Please try again.")
        conv_state_manager.add_message_to_conversation(conversation_id, error_message)
        return {
            "messages": conv_state_manager.get_conversation_messages(conversation_id),
            'conversation_id': conversation_id
        }

    tool_name = last_tool_call["name"]
    tool_args = last_tool_call["args"]

    print(f"[Tools] Executing {tool_name} with args: {tool_args}")

    # Execute cached tool (same as your original implementation)
    result = None
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
            temp_state = {"messages": conversation_messages}
            original_result = retry_tool_call(tool_node, temp_state)
            result = original_result

    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        error_message = AIMessage(content="I encountered an error while searching. Please try again.")
        conv_state_manager.add_message_to_conversation(conversation_id, error_message)
        return {
            "messages": conv_state_manager.get_conversation_messages(conversation_id),
            'conversation_id': conversation_id
        }

    # Create proper tool message
    if tool_name == "query_products" and isinstance(result, dict):
        if result.get("status") == "success" and result.get("results"):
            # Format for better LLM understanding
            formatted_content = f"Found {len(result['results'])} products:\n"
            for i, product in enumerate(result["results"], 1):
                formatted_content += f"{i}. **{product['name']}** - ${product['price']} (Size: {product['size']}, Color: {product['color']})\n"
            tool_content = formatted_content
        else:
            tool_content = json.dumps(result)
    else:
        tool_content = json.dumps(result) if not isinstance(result, str) else result

    tool_message = ToolMessage(
        content=tool_content,
        tool_call_id=last_tool_call["id"]
    )

    # Add tool message to conversation
    conv_state_manager.add_message_to_conversation(conversation_id, tool_message)

    # Handle fallback logic for empty results
    should_fallback = False
    if tool_name == "query_products":
        try:
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result

            if result_data.get("status") == "empty":
                should_fallback = True
                print("[Fallback] Empty query_products result, suggesting semantic search")
        except Exception as e:
            print(f"⚠️ Failed to parse tool response: {e}")

    if should_fallback:
        fallback_instruction = AIMessage(
            content="I couldn't find any products matching those specific criteria. Let me try a semantic search to find similar products that might interest you."
        )
        conv_state_manager.add_message_to_conversation(conversation_id, fallback_instruction)

    return {
        "messages": conv_state_manager.get_conversation_messages(conversation_id),
        'conversation_id': conversation_id
    }


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
        if last_tool_call and last_tool_call["name"] != "semantic_search_tool" or last_tool_call["name"] != "get_product_details":
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
builder.add_node('tools', multi_conv_tools_node)
builder.add_edge(START, 'llm')
builder.add_edge('tools', 'llm')
builder.add_conditional_edges('llm', router, {
    'tools': 'tools',
    'end': END,
})

graph = builder.compile()

# Main interactive loop with cache management
if __name__ == "__main__":
    """Main function that handles multiple conversations correctly"""
    print("🚀 Welcome to your Multi-Conversation E-Commerce Assistant!")
    print("Each session gets its own conversation ID and message history.")
    print("Commands: 'cache_stats', 'conv_stats', 'active_convs', 'switch_conv <id>', 'clear_cache', 'debug', 'quit'")
    print()

    # Start a new conversation
    conversation_id = conversation_manager.start_conversation()
    print(f"📝 Started conversation: {conversation_id[:8]}...\n")

    current_conversation_id = conversation_id

    try:
        while True:
            user_input = input(f"[{current_conversation_id[:8]}] > ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                # End current conversation
                conversation_manager.end_conversation(current_conversation_id)
                conv_state_manager.remove_conversation(current_conversation_id)
                print("Goodbye! 👋")
                break

            elif user_input.lower() == 'debug':
                # Debug information
                conv_messages = conv_state_manager.get_conversation_messages(current_conversation_id)
                print(f"🐛 Debug Info for {current_conversation_id[:8]}:")
                print(f"   Messages in memory: {len(conv_messages)}")
                for i, msg in enumerate(conv_messages[-3:], 1):  # Last 3 messages
                    msg_type = getattr(msg, 'type', 'unknown')
                    content = getattr(msg, 'content', str(msg))
                    print(f"   {i}. [{msg_type}] {content[:50]}...")
                continue

            elif user_input.lower() == 'active_convs':
                active_convs = conv_state_manager.get_active_conversations()
                print(f"🔄 Active Conversations ({len(active_convs)}):")
                for conv in active_convs:
                    conv_id = conv['conversation_id'][:8]
                    last_active = datetime.fromisoformat(conv['last_active']).strftime("%H:%M:%S")
                    current_indicator = "👆" if conv['conversation_id'] == current_conversation_id else "  "
                    print(f"   {current_indicator} {conv_id}... [{last_active}] ({conv['message_count']} msgs)")
                continue

            elif user_input.lower().startswith('switch_conv '):
                new_conv_id = user_input[12:].strip()
                # Find full conversation ID from partial match
                active_convs = conv_state_manager.get_active_conversations()
                matching_conv = None
                for conv in active_convs:
                    if conv['conversation_id'].startswith(new_conv_id) or conv['conversation_id'] == new_conv_id:
                        matching_conv = conv
                        break

                if matching_conv:
                    current_conversation_id = matching_conv['conversation_id']
                    print(f"🔄 Switched to conversation: {current_conversation_id[:8]}...")
                else:
                    print(f"❌ No active conversation found starting with: {new_conv_id}")
                continue

            elif user_input.lower() == 'new_conv':
                # Start a new conversation
                new_conversation_id = conversation_manager.start_conversation()
                current_conversation_id = new_conversation_id
                print(f"📝 Started new conversation: {new_conversation_id[:8]}...")
                continue

            elif user_input.lower() == 'cache_stats':
                stats = cache.get_cache_stats()
                print(f"📊 Cache Stats:")
                print(f"   Active Tool Entries: {stats['tool_cache_active']}")
                print(f"   Expired Tool Entries: {stats['tool_cache_expired']}")
                print(f"   Active LLM Entries: {stats['llm_cache_active']}")
                print(f"   Expired LLM Entries: {stats['llm_cache_expired']}")
                print(f"   Cache Size: {stats['cache_size_bytes'] / 1024:.1f} KB")
                continue

            elif user_input.lower() == 'conv_stats':
                stats = conversation_manager.get_conversation_stats()
                print(f"📈 Conversation Stats:")
                print(f"   Total Conversations: {stats['total_conversations']}")
                print(f"   Active: {stats['active_conversations']}")
                print(f"   Completed: {stats['completed_conversations']}")
                print(f"   Total Messages: {stats['total_messages']}")
                print(f"   Avg Messages/Conversation: {stats['avg_messages_per_conversation']}")
                print(f"   Cache Hit Rate: {stats['cache_hit_rate']}%")
                continue

            print(f"[Main] Processing user input: {user_input}")

            # Save user message to database
            try:
                conversation_manager.save_message(current_conversation_id, 'user', user_input)
            except Exception as e:
                print(f"⚠️ Failed to save user message to DB: {e}")

            # Add user message to conversation-specific state
            user_message = HumanMessage(content=user_input)
            conv_state_manager.add_message_to_conversation(current_conversation_id, user_message)

            print(
                f"[Main] Added message to conversation. Total messages: {len(conv_state_manager.get_conversation_messages(current_conversation_id))}")

            # Create state for this specific conversation
            state = {
                'messages': conv_state_manager.get_conversation_messages(current_conversation_id),
                'conversation_id': current_conversation_id
            }

            print(f"[Main] Created state with {len(state['messages'])} messages")

            # Run the LangGraph pipeline
            start_time = time.time()

            try:
                print("[Main] Invoking multi_conv_graph...")
                state = graph.invoke(state)
                print("[Main] Graph invocation completed")
            except Exception as e:
                print(f"❌ Graph execution error: {e}")
                error_message = AIMessage(content="I encountered an error processing your request. Please try again.")
                conv_state_manager.add_message_to_conversation(current_conversation_id, error_message)
                print("I encountered an error processing your request. Please try again.")
                continue

            end_time = time.time()

            # Print assistant's response
            if state.get('messages') and len(state['messages']) > 0:
                last_message = state['messages'][-1]
                print(f"\n{last_message.content}")
                print(f"⏱️ Response time: {end_time - start_time:.2f}s\n")
            else:
                print("⚠️ No response received from the assistant")

    except KeyboardInterrupt:
        conversation_manager.end_conversation(current_conversation_id)
        conv_state_manager.remove_conversation(current_conversation_id)
        print("\n\nConversation saved. Goodbye! 👋")
    except Exception as e:
        print(f"❌ Unexpected error in main loop: {e}")
        conversation_manager.end_conversation(current_conversation_id)
        conv_state_manager.remove_conversation(current_conversation_id)

