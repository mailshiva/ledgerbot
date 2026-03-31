#!/usr/bin/env python3
"""
TASK 5: Interactive CLI for Credit Card Transactions Agent

Provides a user-friendly REPL for querying financial data with LLM assistance.

Features:
  ✓ Interactive conversation loop (multi-turn)
  ✓ Provider selection (Claude, Ollama, etc.)
  ✓ Session tracking (questions, answers, costs)
  ✓ Exit command handling (Ctrl+C, 'exit', 'quit')
  ✓ Pretty-printed responses
  ✓ Session summary on exit

Usage:
  python main.py chat                    # Default: Claude
  python main.py chat --model qwen3:8b    # Use Ollama
  python main.py chat --provider ollama
  
Interactive:
  > How much did I spend on food in February?
  > What are my top merchants?
  > Show me Walmart transactions
  > exit

Session ends with:
  📊 Session Summary: 3 turns | $0.015 USD | 2.3s total
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.config import LLMConfig, Model
from src.llm.client import LLMClient
from src.agent.agent import Agent
from src.database.db_manager import DatabaseManager


class CLIAgent:
    """CLI wrapper for the Agent with session tracking and pretty printing."""
    
    def __init__(self, provider: str = "gemini", model_name: str = None):
        """
        Initialize CLI agent.
        
        Args:
            provider: "anthropic", "ollama", "gemini", "openai"
            model_name: specific model name or None for provider default
        """
        self.provider = provider.lower()
        self.session_start = datetime.now()
        self.turns = []
        
        # Map provider to model
        if model_name:
            try:
                model = Model[model_name.upper().replace("-", "_")]
            except KeyError:
                print(f"❌ Unknown model: {model_name}")
                self._print_available_models()
                sys.exit(1)
        else:
            model = self._get_default_model()
        
        # Initialize components
        try:
            config = LLMConfig(default_model=model)
            config.validate()  # Will raise if API key missing
            self.llm_client = LLMClient(config)
            self.db = DatabaseManager(str(Path.home() / "sqlLite_DB" / "bank_data.db"))
            self.agent = Agent(db=self.db, llm_client=self.llm_client, max_turns=5)
            
            self.model_name = model.value
            print(f"✓ Agent initialized")
            print(f"  Model: {self.model_name}")
            print(f"  Database: {self.db}")
            
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            sys.exit(1)
    
    def _get_default_model(self) -> Model:
        """Get default model for provider."""
        provider_defaults = {
            "anthropic": Model.CLAUDE_HAIKU,
            "claude": Model.CLAUDE_HAIKU,
            "ollama": Model.QWEN3_8B,
            "gemini": Model.GEMINI_FLASH,
            "openai": Model.GPT4O_MINI,
        }
        return provider_defaults.get(self.provider, Model.GEMINI_FLASH)
    
    def _print_available_models(self):
        """Print available models grouped by provider."""
        print("\nAvailable models:")
        by_provider = {}
        for model in Model:
            provider = model.value.split("-")[0]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model.name)
        
        for provider, models in sorted(by_provider.items()):
            print(f"\n  {provider.upper()}:")
            for m in models:
                print(f"    - {m}")
    
    def chat(self, question: str) -> None:
        """Send a question and get response."""
        try:
            print(f"\n🤔 Processing...\n", end="", flush=True)
            t0 = time.time()
            
            answer = self.agent.chat(question)
            
            elapsed = time.time() - t0
            
            # Get turn info
            turn = self.agent.conversation_history[-1]
            self.turns.append(turn)
            
            # Print answer
            print(answer)
            
            # Print metadata
            print(f"\n  ⏱️  {elapsed:.2f}s | 💰 ${turn.cost_usd:.4f}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def print_session_summary(self) -> None:
        """Print session summary before exit."""
        elapsed = (datetime.now() - self.session_start).total_seconds()
        total_cost = sum(t.cost_usd for t in self.turns)
        total_tokens_in = sum(t.input_tokens for t in self.turns)
        total_tokens_out = sum(t.output_tokens for t in self.turns)
        
        print("\n" + "=" * 80)
        print("📊 SESSION SUMMARY")
        print("=" * 80)
        print(f"Turns:     {len(self.turns)}")
        print(f"Total cost: ${total_cost:.6f} USD")
        print(f"Tokens:    {total_tokens_in:,} input | {total_tokens_out:,} output")
        print(f"Duration:  {elapsed:.1f}s")
        
        if self.turns:
            print(f"\nQuestions asked:")
            for i, turn in enumerate(self.turns, 1):
                q = turn.question[:60] + "..." if len(turn.question) > 60 else turn.question
                print(f"  {i}. {q}")
        
        print("=" * 80)
    
    def run_interactive(self) -> None:
        """Start interactive REPL loop."""
        print("\n" + "=" * 80)
        print("CREDIT CARD TRANSACTIONS — INTERACTIVE AGENT")
        print("=" * 80)
        print(f"\n📝 Ask questions about your spending:")
        print("   'How much did I spend on food in Feb?'")
        print("   'What are my top merchants?'")
        print("   'Show me Walmart transactions'")
        print("\n💡 Commands:")
        print("   'exit', 'quit', 'q' — end session")
        print("   Ctrl+C — interrupt current question")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'exit' to quit.")
            except EOFError:
                # Ctrl+D
                break
        
        self.print_session_summary()
        print("Goodbye! 👋\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Credit Card Transactions - Interactive Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py chat                           # Gemini Flash (default)
  python main.py chat --model ollama            # Ollama qwen3:8b
  python main.py chat --provider anthropic      # CLAUDE_HAIKU
  echo "How much did I spend?" | python main.py chat  # Non-interactive
        """
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="chat",
        help="Command: 'chat' (default), 'test', 'help'"
    )
    
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "claude", "ollama", "gemini"],
        help="LLM provider (default: anthropic)"
    )
    
    parser.add_argument(
        "--model",
        help="Specific model name (e.g., CLAUDE_HAIKU, QWEN3_8B)"
    )
    
    parser.add_argument(
        "--db",
        help="Path to SQLite database (default: ~/sqlLite_DB/bank_data.db)"
    )
    
    args = parser.parse_args()
    
    if args.command == "help":
        parser.print_help()
        sys.exit(0)
    
    if args.command == "test":
        # Quick test
        print("Running quick test...")
        cli = CLIAgent(provider=args.provider, model_name=args.model)
        cli.chat("How much did I spend?")
        cli.print_session_summary()
        sys.exit(0)
    
    if args.command == "chat":
        # Interactive or piped
        cli = CLIAgent(provider=args.provider, model_name=args.model)
        
        # Check if stdin is piped
        if not sys.stdin.isatty():
            # Non-interactive mode: read from stdin
            for line in sys.stdin:
                question = line.strip()
                if question:
                    cli.chat(question)
            cli.print_session_summary()
        else:
            # Interactive mode
            cli.run_interactive()
        
        sys.exit(0)
    
    print(f"Unknown command: {args.command}")
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
