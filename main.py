from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from db import get_all_categories_and_brands

from tools import (
    # list_products_by_category,
    list_brands_by_category,
    # search_products,
    get_product_details,
    # filter_products_by_price,
    # search_by_size_and_category,
    # get_product_images,
    query_products
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
    query_products
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
- If the user asks for more info about a known product, use `get_product_details` with the exact name.
- If you're unsure which brand to search for in a category, call `list_brands_by_category` first.
- Always provide **3 products at most** unless the user asks for more.

---

📊 Available Categories: {', '.join(categories)}
🏷️ Available Brands: {', '.join(brands)}

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

    if getattr(last_message, 'tool_calls', None):
        return 'tools'  # product-related, call tools
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
    query_products
])

# Tools node: executes tool calls and returns updated messages
def tools_node(state):
    result = tool_node.invoke(state)
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
        state['messages'].append({'role': 'user', 'content': user_input})

        # Run the LangGraph pipeline (LLM + tools)
        state = graph.invoke(state)

        # Print assistant's response
        print(state['messages'][-1].content, "\n")

# from db import fetch_all_products
# from tools import embed_products, semantic_search, build_sentence
#
# print("🔄 Fetching product data from database...")
# products = fetch_all_products()
#
# print("🧠 Generating sentence embeddings...")
# embedded_products = embed_products(products)
#
# while True:
#     query = input("\n💬 Ask me for a product (or 'quit'): ")
#     if query.lower() == "quit":
#         break
#
#     print("🔍 Searching...")
#     results = semantic_search(query, embedded_products)
#     for idx, r in enumerate(results, 1):
#         print(f"\n#{idx}: {r['product_name']} ({r['brand']})")
#         print(f"Category: {r['category']} | Gender: {r['gender']} | Price: ${r['price']}")
#         print(f"Score: {r['score']:.4f}")
#         print(f"→ {build_sentence(r)}")

