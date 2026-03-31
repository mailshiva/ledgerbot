#!/usr/bin/env python3
"""
Standalone Ollama Agent for Bank Statement Queries

Run this independently using local Ollama (qwen3:8b or other models).
No API keys needed, fully local.

Requirements:
    - Ollama running: ollama serve
    - Model pulled: ollama pull qwen3:8b

Usage:
    python standalone_ollama_agent.py "How much did I spend on food?"
"""

import json
import urllib.request
import urllib.error
from pathlib import Path
import sqlite3

# Load database
DB_PATH = Path.home() / "sqlLite_DB/bank_data.db"
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

def query_transactions(sql: str) -> list:
    """Execute SQL query on transactions table"""
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_categories() -> list:
    """Get spending by category"""
    try:
        sql = """
        SELECT category, 
               SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as total_spent,
               COUNT(*) as transaction_count
        FROM transactions
        WHERE category != 'Uncategorized'
        GROUP BY category
        ORDER BY total_spent DESC
        """
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_spending_summary() -> dict:
    """Get overall spending summary"""
    try:
        sql = """
        SELECT 
            SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as total_spent,
            SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END) as total_received,
            COUNT(*) as total_transactions,
            COUNT(DISTINCT strftime('%Y-%m', date)) as months_of_data
        FROM transactions
        """
        cursor = conn.execute(sql)
        row = cursor.fetchone()
        return dict(row) if row else {}
    except Exception as e:
        return {"error": str(e)}

def call_ollama(prompt: str) -> str:
    """Call Ollama API with streaming response"""
    
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1,
    }).encode("utf-8")
    
    try:
        req = urllib.request.Request(
            OLLAMA_API,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
            
    except urllib.error.URLError as e:
        return f"Error connecting to Ollama at {OLLAMA_API}: {e}\nMake sure Ollama is running: ollama serve"
    except Exception as e:
        return f"Error: {e}"

def analyze_question(question: str) -> str:
    """
    Analyze user question and determine what data to fetch.
    Uses simple heuristics to route to appropriate data queries.
    """
    
    question_lower = question.lower()
    
    # Check for specific queries
    if any(word in question_lower for word in ["category", "categories", "breakdown", "by type"]):
        print("  📊 Getting spending by category...")
        categories = get_categories()
        data = json.dumps(categories, indent=2)
        return f"Spending by category:\n{data}"
    
    elif any(word in question_lower for word in ["summary", "total", "overall", "in total"]):
        print("  📊 Getting spending summary...")
        summary = get_spending_summary()
        data = json.dumps(summary, indent=2)
        return f"Spending summary:\n{data}"
    
    elif any(word in question_lower for word in ["walmart", "costco", "amazon", "food", "grocery"]):
        # Try to extract merchant or category from question
        if "walmart" in question_lower or "sam" in question_lower:
            sql = """
            SELECT date, description, amount, merchant_name, category
            FROM transactions
            WHERE LOWER(merchant_name) LIKE '%walmart%' OR LOWER(merchant_name) LIKE '%sams%'
            ORDER BY date DESC LIMIT 20
            """
        elif "food" in question_lower or "dining" in question_lower or "grocery" in question_lower:
            sql = """
            SELECT date, description, amount, merchant_name, category
            FROM transactions
            WHERE category IN ('Food & Dining', 'Groceries', 'Restaurants')
            ORDER BY date DESC LIMIT 20
            """
        else:
            sql = "SELECT date, description, amount, merchant_name, category FROM transactions LIMIT 20"
        
        print("  📋 Querying transactions...")
        data = json.dumps(query_transactions(sql), indent=2, default=str)
        return f"Transaction data:\n{data}"
    
    else:
        # Default: get recent transactions
        print("  📋 Fetching recent transactions...")
        sql = "SELECT date, description, amount, merchant_name, category FROM transactions ORDER BY date DESC LIMIT 10"
        data = json.dumps(query_transactions(sql), indent=2, default=str)
        return f"Recent transactions:\n{data}"

def agent_chat(question: str) -> str:
    """
    Process question with Ollama agent.
    
    1. Analyze question to determine what data to fetch
    2. Fetch relevant transaction data
    3. Send to Ollama with context
    4. Return answer
    """
    
    print("\n🔍 Processing question...")
    
    # Get relevant data
    data_context = analyze_question(question)
    
    # Build prompt for Ollama
    system_prompt = """You are a helpful financial advisor. Answer questions about bank transactions.
Use the transaction data provided to give specific, accurate answers with numbers.
If asked a question you can't answer from the data, say so clearly."""
    
    prompt = f"""{system_prompt}

Transaction data available:
{data_context}

User question: {question}

Answer the question based on the transaction data above:"""
    
    print("  🤔 Asking Ollama...")
    answer = call_ollama(prompt)
    
    return answer

def main():
    """Interactive chat loop"""
    
    print("=" * 80)
    print("OLLAMA BANK STATEMENT AGENT (Standalone - Local)")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    print("API: http://localhost:11434")
    print("\nAsk questions about your spending:")
    print("  'How much did I spend on food?'")
    print("  'What's my total spending?'")
    print("  'Show me Walmart transactions'")
    print("\nType 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            print("\nAgent: ", end="", flush=True)
            answer = agent_chat(user_input)
            print(answer)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        question = " ".join(sys.argv[1:])
        print(f"\nQuestion: {question}")
        answer = agent_chat(question)
        print(f"\nAnswer:\n{answer}")
    else:
        # Interactive mode
        main()
