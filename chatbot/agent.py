"""LangChain agent with Gemini for chatbot functionality."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot.tools import all_tools
from config.settings import settings

load_dotenv()


# =============================================================================
# Debug Logging Configuration
# =============================================================================

# Create logger for agent debugging
agent_logger = logging.getLogger("chatbot.agent")
agent_logger.setLevel(logging.DEBUG)

# Create file handler for debug logs
_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "agent_debug.log")

_file_handler = logging.FileHandler(_log_file, mode="a", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
agent_logger.addHandler(_file_handler)

# Console handler for development (optional, can be disabled)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(
    logging.Formatter("🤖 %(message)s")
)
agent_logger.addHandler(_console_handler)


@dataclass
class ToolExecutionLog:
  """Record of a single tool execution."""
  tool_name: str
  arguments: Dict[str, Any]
  result: str
  success: bool
  execution_time_ms: float
  timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
  error: Optional[str] = None


@dataclass
class ConversationLog:
  """Record of a full conversation turn."""
  conversation_id: str
  user_message: str
  timestamp: str
  tool_executions: List[ToolExecutionLog] = field(default_factory=list)
  final_response: str = ""
  total_time_ms: float = 0
  iterations: int = 0


# In-memory log storage for recent conversations (accessible from UI)
_conversation_logs: List[ConversationLog] = []
_max_logs = 50  # Keep last 50 conversations


def get_recent_logs(limit: int = 10) -> List[Dict]:
  """Get recent conversation logs for debugging."""
  return [asdict(log) for log in _conversation_logs[-limit:]]


def get_last_log() -> Optional[Dict]:
  """Get the most recent conversation log."""
  if _conversation_logs:
    return asdict(_conversation_logs[-1])
  return None


def clear_logs():
  """Clear all in-memory logs."""
  global _conversation_logs
  _conversation_logs = []
  agent_logger.info("Logs cleared")


def _log_tool_execution(log: ToolExecutionLog):
  """Log a tool execution to both file and memory."""
  status = "✓" if log.success else "✗"
  args_str = json.dumps(log.arguments, ensure_ascii=False, default=str)

  # Log full details to file
  agent_logger.debug(
      f"TOOL [{status}] {log.tool_name} | "
      f"Time: {log.execution_time_ms:.1f}ms"
  )
  agent_logger.debug(f"  ARGS: {args_str}")
  agent_logger.debug(f"  RESULT:\n{log.result}")

  if not log.success:
    agent_logger.error(f"TOOL ERROR {log.tool_name}: {log.error}")


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful data analyst assistant for the Bangkok Urban Reports (Traffy Fondue) visualization dashboard.

You help users explore and analyze urban problem reports from Bangkok citizens. The data includes reports about:
- Roads (ถนน), sidewalks (ทางเท้า), bridges (สะพาน)
- Flooding (น้ำท่วม), drainage (ท่อระบายน้ำ), canals (คลอง)
- Traffic (จราจร), traffic signs (ป้ายจราจร), obstructions (กีดขวาง)
- Cleanliness (ความสะอาด), trash, sanitation
- Safety (ความปลอดภัย), lighting (แสงสว่าง)
- Trees (ต้นไม้), stray animals (สัตว์จรจัด), homeless (คนจรจัด)
- Noise (เสียงรบกวน), PM2.5, electrical wires (สายไฟ)
- And more categories...

## Your Capabilities

### 1. Data Analysis Tools (use these to answer analytical questions)

- **get_data_schema**: ALWAYS call this first when unsure about available data. Shows columns, dimensions, sample values.
- **get_ticket_counts**: Count/group tickets by dimension (type, district, status, org). Best for "which X has the most Y" questions.
- **get_statistics**: Get summary stats (totals, completion rates, resolution times). Good for overview questions.
- **get_time_series**: Analyze trends over time. Use for "how has X changed" or "trend of Y" questions.
- **get_crosstab**: Cross-tabulate two dimensions. Use for "which district has most flooding" type questions.
- **run_analytical_query**: Flexible query builder for complex questions that don't fit other tools.
- **get_ticket_detail**: Get details about a specific ticket ID.

### 2. Visualization Controls

- **set_visualization_layer**: Switch between scatter, heatmap, hexagon, cluster views
- **filter_categories**: Show only specific report types on the map
- **zoom_to_district**: Focus map on a specific Bangkok district
- **highlight_tickets_on_map**: Highlight specific tickets visually
- **set_layer_opacity/set_point_radius**: Adjust visual appearance
- **reset_all_filters**: Reset to show all data

## Strategy for Answering Questions

1. **For analytical questions** (e.g., "What ticket type has the highest reoccurrence?"):
   - Use get_ticket_counts with appropriate group_by and order_by
   - Example: get_ticket_counts(group_by="type", order_by="count_desc", limit=10)

2. **For comparison questions** (e.g., "Which districts have most flooding?"):
   - Use get_crosstab or get_ticket_counts with filters
   - Example: get_ticket_counts(group_by="district", filters='{"type": "น้ำท่วม"}')

3. **For trend questions** (e.g., "How have reports changed this month?"):
   - Use get_time_series with appropriate granularity
   - Example: get_time_series(granularity="day", days_back=30)

4. **For statistics** (e.g., "What's the completion rate?"):
   - Use get_statistics

5. **For visualization requests** (e.g., "Show me the heatmap"):
   - Use the UI control tools

## Guidelines

- Be data-driven: query first, then present findings
- Be concise: summarize results, highlight key insights
- If users ask in Thai, respond in Thai
- For ambiguous questions, call get_data_schema to understand what's queryable
- Format numbers with commas for readability
- When showing results, explain what they mean

Available dimensions for grouping/filtering: type, district, status, org, province
Available time granularities: hour, day, week, month, year
Available map layers: scatter, heatmap, hexagon, cluster
"""


def get_llm() -> ChatGoogleGenerativeAI:
  """Get configured Gemini LLM instance."""
  api_key = settings.gemini.api_key
  if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please set it in your .env file."
    )

  return ChatGoogleGenerativeAI(
      model=settings.gemini.model,
      google_api_key=api_key,
      temperature=0.7,
      streaming=True,
      max_retries=0,  # Don't auto-retry on rate limits
  )


def get_agent():
  """
  Create and return the LangChain agent with tools.

  Returns:
      Configured agent with Gemini and MCP tools
  """
  llm = get_llm()

  # Bind tools to the LLM
  llm_with_tools = llm.bind_tools(all_tools)

  return llm_with_tools


def create_prompt_messages(
    user_message: str,
    chat_history: Optional[List[dict]] = None,
) -> List:
  """
  Create the message list for the agent.

  Args:
      user_message: The current user message
      chat_history: Optional list of previous messages

  Returns:
      List of messages for the LLM
  """
  messages = [SystemMessage(content=SYSTEM_PROMPT)]

  # Add chat history
  if chat_history:
    for msg in chat_history[-10:]:  # Keep last 10 messages for context
      role = msg.get("role", "user")
      content = msg.get("content", "")
      if role == "user":
        messages.append(HumanMessage(content=content))
      else:
        messages.append(AIMessage(content=content))

  # Add current message
  messages.append(HumanMessage(content=user_message))

  return messages


def invoke_agent_stream(
    user_message: str,
    chat_history: Optional[List[dict]] = None,
) -> Generator[str, None, dict]:
  """
  Invoke the agent and stream the response.

  This function handles tool calls automatically and streams text responses.

  Args:
      user_message: The user's message
      chat_history: Optional list of previous chat messages

  Yields:
      String chunks of the response

  Returns:
      Dict with final response info including any tool calls made
  """
  from langchain_core.messages import ToolMessage
  import uuid

  # Start conversation logging
  conversation_start = time.time()
  conversation_id = str(uuid.uuid4())[:8]
  conversation_log = ConversationLog(
      conversation_id=conversation_id,
      user_message=user_message,
      timestamp=datetime.now().isoformat(),
  )

  agent_logger.info(f"═══ CONVERSATION START [{conversation_id}] ═══")
  agent_logger.info(f"USER: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")

  llm = get_llm()
  llm_with_tools = llm.bind_tools(all_tools)

  messages = create_prompt_messages(user_message, chat_history)
  tool_calls_made = []
  final_response = ""

  # Maximum iterations to prevent infinite loops
  max_iterations = 5
  iteration = 0

  while iteration < max_iterations:
    iteration += 1
    agent_logger.debug(f"[{conversation_id}] Iteration {iteration}/{max_iterations}")

    # Stream the response
    response_content = ""
    tool_calls = []

    for chunk in llm_with_tools.stream(messages):
      # Accumulate content
      if chunk.content:
        response_content += chunk.content
        yield chunk.content

      # Collect tool calls
      if hasattr(chunk, "tool_calls") and chunk.tool_calls:
        tool_calls.extend(chunk.tool_calls)

    # If no tool calls, we're done
    if not tool_calls:
      final_response = response_content
      agent_logger.debug(f"[{conversation_id}] No more tool calls, finishing")
      break

    # Process tool calls
    agent_logger.info(f"[{conversation_id}] Model requested {len(tool_calls)} tool(s)")
    messages.append(AIMessage(content=response_content, tool_calls=tool_calls))

    for tool_call in tool_calls:
      tool_name = tool_call.get("name", "")
      tool_args = tool_call.get("args", {})
      tool_id = tool_call.get("id", "")

      # Execute tool with timing and logging
      tool_result, tool_log = _execute_tool_with_logging(tool_name, tool_args)
      conversation_log.tool_executions.append(tool_log)

      tool_calls_made.append({
          "name": tool_name,
          "args": tool_args,
          "result": tool_result,
      })

      # Add tool result to messages
      messages.append(
          ToolMessage(content=tool_result, tool_call_id=tool_id)
      )

      # Yield status indicator
      yield f"\n[Used {tool_name}]\n"

  # Finalize conversation log
  conversation_log.final_response = final_response[:500]
  conversation_log.total_time_ms = (time.time() - conversation_start) * 1000
  conversation_log.iterations = iteration

  # Store log
  _conversation_logs.append(conversation_log)
  if len(_conversation_logs) > _max_logs:
    _conversation_logs.pop(0)

  # Log summary
  tool_count = len(conversation_log.tool_executions)
  agent_logger.info(
      f"═══ CONVERSATION END [{conversation_id}] ═══ "
      f"Tools: {tool_count} | Time: {conversation_log.total_time_ms:.0f}ms | Iterations: {iteration}"
  )

  return {
      "response": final_response,
      "tool_calls": tool_calls_made,
      "debug_log": asdict(conversation_log),
  }


def _execute_tool_with_logging(tool_name: str, tool_args: dict) -> tuple[str, ToolExecutionLog]:
  """
  Execute a tool with full debug logging.

  Args:
      tool_name: Name of the tool to execute
      tool_args: Arguments for the tool

  Returns:
      Tuple of (result string, ToolExecutionLog)
  """
  start_time = time.time()
  tool_map = {tool.name: tool for tool in all_tools}

  # Log the call
  args_preview = json.dumps(tool_args, ensure_ascii=False, default=str)[:300]
  agent_logger.info(f"→ CALLING: {tool_name}")
  agent_logger.debug(f"  Args: {args_preview}")

  if tool_name not in tool_map:
    error_msg = f"Error: Unknown tool '{tool_name}'"
    log = ToolExecutionLog(
        tool_name=tool_name,
        arguments=tool_args,
        result=error_msg,
        success=False,
        execution_time_ms=0,
        error=error_msg,
    )
    _log_tool_execution(log)
    return error_msg, log

  try:
    tool = tool_map[tool_name]
    result = tool.invoke(tool_args)
    result_str = str(result)

    execution_time = (time.time() - start_time) * 1000

    log = ToolExecutionLog(
        tool_name=tool_name,
        arguments=tool_args,
        result=result_str,
        success=True,
        execution_time_ms=execution_time,
    )
    _log_tool_execution(log)

    # Log success with full result to file
    agent_logger.info(f"← RESULT: {tool_name} ({execution_time:.0f}ms) - {len(result_str)} chars")

    return result_str, log

  except Exception as e:
    execution_time = (time.time() - start_time) * 1000
    error_msg = f"Error executing {tool_name}: {str(e)}"

    log = ToolExecutionLog(
        tool_name=tool_name,
        arguments=tool_args,
        result=error_msg,
        success=False,
        execution_time_ms=execution_time,
        error=str(e),
    )
    _log_tool_execution(log)

    agent_logger.error(f"✗ FAILED: {tool_name} - {str(e)}")

    return error_msg, log


def _execute_tool(tool_name: str, tool_args: dict) -> str:
  """
  Execute a tool by name with given arguments (simple version).

  Args:
      tool_name: Name of the tool to execute
      tool_args: Arguments for the tool

  Returns:
      Tool execution result as string
  """
  result, _ = _execute_tool_with_logging(tool_name, tool_args)
  return result


def invoke_agent_simple(
    user_message: str,
    chat_history: Optional[List[dict]] = None,
) -> str:
  """
  Invoke the agent without streaming (simpler interface).

  Args:
      user_message: The user's message
      chat_history: Optional chat history

  Returns:
      Complete response string
  """
  chunks = []
  for chunk in invoke_agent_stream(user_message, chat_history):
    chunks.append(chunk)
  return "".join(chunks)


# =============================================================================
# Debug Log Formatting & Utilities
# =============================================================================


def format_debug_log(log: Optional[Dict] = None) -> str:
  """
  Format a debug log for human-readable display.

  Args:
      log: Optional specific log dict, defaults to last log

  Returns:
      Formatted string representation of the log
  """
  if log is None:
    log = get_last_log()

  if not log:
    return "No debug logs available."

  lines = [
      "╔══════════════════════════════════════════════════════════════╗",
      f"║ DEBUG LOG: Conversation {log.get('conversation_id', 'N/A')}",
      f"║ Time: {log.get('timestamp', 'N/A')}",
      "╠══════════════════════════════════════════════════════════════╣",
      f"║ USER INPUT:",
      f"║   {log.get('user_message', 'N/A')[:60]}...",
      "╠══════════════════════════════════════════════════════════════╣",
      f"║ TOOL EXECUTIONS ({len(log.get('tool_executions', []))} total):",
  ]

  for i, tool_exec in enumerate(log.get("tool_executions", []), 1):
    status = "✓" if tool_exec.get("success") else "✗"
    lines.extend([
        f"║ ",
        f"║ {i}. [{status}] {tool_exec.get('tool_name', 'N/A')}",
        f"║    Time: {tool_exec.get('execution_time_ms', 0):.1f}ms",
        f"║    Args: {json.dumps(tool_exec.get('arguments', {}), ensure_ascii=False, default=str)[:50]}...",
    ])

    result = tool_exec.get("result", "")[:100].replace("\n", " ")
    lines.append(f"║    Result: {result}...")

    if tool_exec.get("error"):
      lines.append(f"║    ERROR: {tool_exec.get('error')}")

  lines.extend([
      "╠══════════════════════════════════════════════════════════════╣",
      f"║ SUMMARY:",
      f"║   Total Time: {log.get('total_time_ms', 0):.0f}ms",
      f"║   Iterations: {log.get('iterations', 0)}",
      f"║   Tools Called: {len(log.get('tool_executions', []))}",
      "╚══════════════════════════════════════════════════════════════╝",
  ])

  return "\n".join(lines)


def get_log_file_path() -> str:
  """Get the path to the debug log file."""
  return _log_file


def read_log_file(lines: int = 100) -> str:
  """
  Read the last N lines from the log file.

  Args:
      lines: Number of lines to read

  Returns:
      Log file contents
  """
  try:
    with open(_log_file, "r", encoding="utf-8") as f:
      all_lines = f.readlines()
      return "".join(all_lines[-lines:])
  except FileNotFoundError:
    return "Log file not found."
  except Exception as e:
    return f"Error reading log file: {e}"
