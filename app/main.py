from langchain_core.messages import HumanMessage

from app.graph import build_graph
from app.tools import init_vectorstore


def main():
    init_vectorstore(force_rebuild=False)

    graph = build_graph()

    while True:
        user_input = input("\nAsk a question (or 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "route": "",
            "retrieved_docs": [],
            "web_results": [],
            "final_answer": "",
            "sql_query": user_input,
            "python_code": "",
            "approval_required": False,
            "approval_message": "",
            "approved": None,
            "status": "new",
            "judge_score": 0.0,
            "retry_count": 0,
            "error": None,
        }

        result = graph.invoke(state)

        if result.get("approval_required"):
            print("\n" + result["approval_message"])
            answer = input("Approve? [y/n]: ").strip().lower()
            if answer == "y":
                if "SQL" in result["approval_message"]:
                    state["sql_query"] = user_input
                    state["approved"] = True
                    result = graph.invoke({
                        **state,
                        "approved": True,
                        "approval_required": False,
                    })
                elif "Python" in result["approval_message"]:
                    state["python_code"] = user_input
                    state["approved"] = True
                    result = graph.invoke({
                        **state,
                        "approved": True,
                        "approval_required": False,
                    })
            else:
                result["final_answer"] = "Execution was denied by the human reviewer."
                print("\n" + result["final_answer"])
                continue

        print("\nAnswer:")
        print(result.get("final_answer", "No answer generated."))


if __name__ == "__main__":
    main()
