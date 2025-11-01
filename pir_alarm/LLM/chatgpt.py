from openai import OpenAI
import os, json, requests
from datetime import datetime

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- OpenAI Client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load historical data for RAG ---
with open("pir_data.json") as f:
    data = json.load(f)

docs = [
    Document(
        page_content=(
            f"Device: {d['devid']}, Timestamp: {d['timestamp']}, "
            f"Anomaly Score: {d.get('score', 'N/A')}"
        ),
        metadata=d
    )
    for d in data
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# --- RAG Chain using LCEL (LangChain 1.0+) ---
combine_prompt = ChatPromptTemplate.from_template(
    "You are an IoT historian. Use only the following retrieved sensor logs to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n"
    "Answer concisely and factually."
)

combine_docs_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | combine_prompt
    | llm
    | StrOutputParser()
)

# Final RAG chain: input -> retrieve -> combine -> answer (string)
rag_chain = combine_docs_chain

# --- Helper for live data ---
def fetch_live_data(device_id):
    return {
        "devid": device_id,
        "timestamp": "2025-10-28 02:13:00",
        "duration": "3s",
        "luminance": 10,
        "score": 0.12
    }

# --- Intent Parser ---
def parse_intent(user_input: str):
    prompt = f"""
You are an intent parser for a smart-home IoT chatbot.

Classify this user question:
"{user_input}"

Return **only valid JSON** with:
- intent: "api", "rag", or "both"
- device_id: exact device ID if mentioned (like 70:2c:1f:37:c3:b6), else null
- time_range: if time mentioned (e.g., 'today', 'last night', 'past week'), else null
- query: short version of the question

Example:
{{"intent": "rag", "device_id": "70:2c:1f:37:c3:b6", "time_range": "night", "query": "why trigger at night"}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only JSON. No explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print("Intent parsing failed:", e)
        return {"intent": "rag", "device_id": None, "time_range": None, "query": user_input}

# --- Main Handler ---
def handle_user_query(user_input: str):
    intent_data = parse_intent(user_input)
    intent = intent_data.get("intent", "rag")
    device_id = intent_data.get("device_id")
    time_range = intent_data.get("time_range")
    print("Intent parsed:", intent_data)

    context_parts = []

    # --- 1. Live API Data ---
    if intent in ("api", "both") and device_id:
        try:
            api_data = fetch_live_data(device_id)
            context_parts.append(f"**Live Status ({device_id})**:\n"
                                 f"Time: {api_data['timestamp']}\n"
                                 f"Motion: {api_data['motion']}, Temp: {api_data['temperature']}°C, "
                                 f"Humidity: {api_data['humidity']}%, Luminance: {api_data['luminance']}\n"
                                 f"Anomaly Score: {api_data['score']}")
        except Exception as e:
            context_parts.append(f"Live data error: {str(e)}")

    # --- 2. Historical RAG Insight ---
    if intent in ("rag", "both"):
        try:
            # Optional: enhance query with time_range
            enhanced_input = user_input
            if time_range and device_id:
                enhanced_input = f"{user_input} (focus on {time_range} for device {device_id})"

            rag_answer = rag_chain.invoke(enhanced_input)
            context_parts.append(f"**Historical Pattern**:\n{rag_answer}")
        except Exception as e:
            context_parts.append(f"Historical lookup failed: {str(e)}")

    # --- Build Final Context ---
    full_context = "\n\n".join(context_parts).strip()
    if not full_context:
        full_context = "No context available."

    # --- Final LLM Answer ---
    final_prompt = f"""
You are a friendly IoT assistant for motion sensors (PIR).

User Question: {user_input}

Context:
{full_context}

Answer naturally, clearly, and helpfully. Explain triggers, patterns, and suggestions.
"""

    try:
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a smart home expert. Be concise and practical."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.6,
            max_tokens=500
        )
        return final.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- Test ---
if __name__ == "__main__":
    print("="*60)
    print(handle_user_query("Why does device 70:2c:1f:37:c3:b6 always trigger at night?"))
    print("\n" + "="*60)
    print(handle_user_query("Show me today's alarms for device 70:2c:1f:37:c3:b6"))