import uuid
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.types import interrupt, Command
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Sreeni's LangGraph Expense Approval API",
    version="1.0.0",
    contact={
        "name": "Sreeni",
        "email": "sreeniusa@outlook.com",
    }
)

# Initialize the Azure OpenAI LLM for conversation-based tasks
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


class ExpenseRequest(BaseModel):
    """
    Pydantic model for receiving expense data.

    Attributes:
        employee_name (str): Name of the employee submitting the expense.
        expense_amount (float): Amount of the expense.
        expense_reason (str): Reason for the expense.
    """
    employee_name: str
    expense_amount: float
    expense_reason: str


class State(TypedDict):
    """
    Graph state containing expense details and approval status.

    Attributes:
        request_id (str): Unique identifier for the expense request.
        employee_name (str): Name of the employee submitting the expense.
        expense_amount (float): Amount of the expense.
        expense_reason (str): Reason for the expense.
        approval_status (list[str]): List of approval statuses and comments.
    """
    request_id: str
    employee_name: str
    expense_amount: float
    expense_reason: str
    approval_status: Annotated[list[str], add_messages]


def review_expense(state: State):
    """
    Initial AI-based review of the expense request.

    Args:
        state (State): Current state of the expense request.

    Returns:
        dict or Command: Updated state or command to move to the next node.
    """
    print("\n[review_expense] Reviewing the expense request...")
    expense_amount = state["expense_amount"]

    if expense_amount <= 50:
        approval_status = "Auto Approved"
        print(f"[review_expense] Approval Status: {approval_status}\n")

        state["approval_status"].append(approval_status)
        return Command(update={"approval_status": state["approval_status"]}, goto="end_node")

    approval_status = "Needs Human Review"
    print(f"[review_expense] Approval Status: {approval_status}\n")
    state["approval_status"].append(approval_status)
    return {"approval_status": state["approval_status"]}


def human_approval_node(state: State):
    """
    Human intervention node for approving or rejecting the expense request.

    Args:
        state (State): Current state of the expense request.

    Returns:
        Command: Command to update the state and move to the next node.
    """
    print("\n[human_approval_node] Awaiting human approval...")
    approval_status = state["approval_status"]

    user_feedback = interrupt(
        {"approval_status": approval_status, "message": "Approve, Reject, or provide comments."})
    print(f"[human_approval_node] Received human feedback: {user_feedback}")

    if user_feedback.lower() in ["approve", "approved"]:
        return Command(update={"approval_status": state["approval_status"] + ["Final Approved"]}, goto="end_node")
    elif user_feedback.lower() in ["reject", "rejected"]:
        return Command(update={"approval_status": state["approval_status"] + ["Final Rejected"]}, goto="end_node")

    return Command(update={"approval_status": state["approval_status"] + [user_feedback]}, goto="review_expense")


def end_node(state: State):
    """
    Final node in the approval process.

    Args:
        state (State): Current state of the expense request.

    Returns:
        dict: Final approval status.
    """
    print("\n[end_node] Process finished.")
    print("Final Approval Status:", state["approval_status"][-1])
    return {"approval_status": state["approval_status"]}


# Building the Graph
graph_builder = StateGraph(State)
graph_builder.add_node("review_expense", review_expense)
graph_builder.add_node("human_approval_node", human_approval_node)
graph_builder.add_node("end_node", end_node)

# Define the Flow
graph_builder.add_edge(START, "review_expense")
graph_builder.add_edge("review_expense", "human_approval_node")
graph_builder.add_edge("human_approval_node", "review_expense")
graph_builder.add_edge("human_approval_node", "end_node")

# Set the finish point
graph_builder.set_finish_point("end_node")

# Enable Interrupt Mechanism
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Save the graph as an image (optional)
image = graph.get_graph().draw_mermaid_png()
with open("expense_approval_graph.png", "wb") as file:
    file.write(image)


@app.post("/submit-expense/")
@app.post("/submit-expense/")
def submit_expense(expense: ExpenseRequest):
    """
    API endpoint to submit an expense request.

    Args:
        expense (ExpenseRequest): Expense request data.

    Returns:
        dict: Request ID and final approval status.
    """

    # Print all values of the ExpenseRequest
    print("\n--- Expense Request Details ---")
    print(f"Employee Name: {expense.employee_name}")
    print(f"Expense Amount: ${expense.expense_amount:.2f}")
    print(f"Expense Reason: {expense.expense_reason}")
    print("-------------------------------\n")

    request_id = str(uuid.uuid4())

    initial_state = {
        "request_id": request_id,
        "employee_name": expense.employee_name,
        "expense_amount": expense.expense_amount,
        "expense_reason": expense.expense_reason,
        "approval_status": []
    }

    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    final_state = initial_state

    for chunk in graph.stream(initial_state, config=thread_config):
        for node_id, value in chunk.items():
            if node_id == "review_expense":
                final_state.update(value)
                if "Auto Approved" in final_state["approval_status"]:
                    print("[submit_expense] Auto-approved. Skipping human approval.")
                    return {"request_id": request_id, "approval_status": final_state["approval_status"]}
            elif node_id == "__interrupt__":
                while True:
                    user_feedback = input("Approve, Reject, or provide comments: ")
                    final_state["approval_status"] = final_state.get("approval_status", []) + [user_feedback]
                    graph.invoke(Command(resume=user_feedback), config=thread_config)
                    if user_feedback.lower() in ["approve", "approved", "reject", "rejected"]:
                        break
            else:
                final_state.update(value)

    return {"request_id": request_id, "approval_status": final_state["approval_status"]}


# Run FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
