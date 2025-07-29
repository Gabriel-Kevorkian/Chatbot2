from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from db import get_all_categories_and_brands
from langchain_core.messages import HumanMessage

from tools import (
    # list_products_by_category,
    list_brands_by_category,
    # search_products,
    get_product_details,
    # filter_products_by_price,
    # search_by_size_and_category,
    # get_product_images,
    query_products,
    semantic_search_tool
)

# Initialize the LLM model
llm = init_chat_model("qwen2.5:7b-instruct-q4_K_M", model_provider="ollama")

# Bind all tools to the LLM
llm = llm.bind_tools([
    # list_products_by_category,
    list_brands_by_category,
    # search_products,
    get_product_details,
    # filter_products_by_price,
    # search_by_size_and_category,
    # get_product_images,
    query_products,
    semantic_search_tool
])


# Define the state schema
class ChatState(TypedDict):
    messages: list

# LLM node: invokes the LLM on conversation messages
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
4. semantic_search(query, top_k=3)
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


- You must **always use tools** for product information. Never guess or generate product data.
- If the user's message is ambiguous (e.g., "show me shirts"), use `query_products` with just the category.
- If the user's message is ambiguous or vague (e.g., "show me something nice"), call `semantic_search`.
- If the user asks for more info about a known product, use `get_product_details` with the exact name.
- If you're unsure which brand to search for in a category, call `list_brands_by_category` first.
- Always provide **3 products at most** unless the user asks for more.

---

📊 Available Categories: {', '.join(categories)}
🏷️ Available Brands: {', '.join(brands)}

IMPORTANT: You must NOT generate or infer any product information on your own.  
You can only repeat and explain the data that comes directly from the tools.  
If the tools return no data, respond only with:  
"I'm sorry, I couldn't find any matching products in our database."
Do NOT make up product names, descriptions, or prices under any circumstances.

Your goal is to use tools correctly and only reply based on what the tools return.
"""


def llm_node(state):
    system_message = {
        'role': 'system',
        'content': system_message_content
    }
    messages_with_system = [system_message] + state['messages']
    response = llm.invoke(messages_with_system)
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


# ToolNode setup with all your tools
tool_node = ToolNode([
    # list_products_by_category,
    list_brands_by_category,
    # search_products,
    get_product_details,
    # filter_products_by_price,
    # search_by_size_and_category,
    # get_product_images,
    query_products,
    semantic_search_tool
])

# Tools node: executes tool calls and returns updated messages
def tools_node(state):
    print("[Tools Node] Invoked")
    result = tool_node.invoke(state)
    print("[Tools Node] Result messages count:", len(result['messages']))
    return {'messages': state['messages'] + result['messages']}
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

# Main interactive loop
if __name__ == "__main__":
    state = {'messages': []}
    print("Welcome to your E-Commerce Assistant!")
    print("Type your question or 'quit' to exit.\n")

    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Add user message to the conversation state
        state['messages'].append(HumanMessage(content=user_input))

        # Run the LangGraph pipeline (LLM + tools)
        state = graph.invoke(state)

        # Print assistant's response
        print(state['messages'][-1].content, "\n")


