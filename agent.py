
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END 
from daytona_utils import create_models, sandbox, download_charts
from prompts import Generator_PROMPT, Reflector_PROMPT, Summarizer_PROMPT, Manager_PROMPT
from schema import GeneratorOutput, ReflectorOutput, AgentState, ManagerOutput
from typing import cast
from dotenv import load_dotenv
from langgraph.types import Command
from typing import Literal

load_dotenv()

MANAGER, GENERATOR, REFLECTOR, SUMMARIZER = create_models({"model": "gpt-4.1-mini"})
  
def manager_cmd(state: AgentState) -> Command[Literal["code_gen", "summarizer","__end__"]]:

    resp = cast(ManagerOutput, MANAGER.invoke([Manager_PROMPT] + state["messages"]))
    decision = resp.decision

    if decision == "code_gen":
        return Command(update = {"question": resp.question},
                   goto = "code_gen"
 )
    elif decision == "summarizer":
        return Command(update = {"question": resp.question},
                   goto = "summarizer"
 )  
    else:
        return Command(update = {"messages": [AIMessage(content=resp.messages)]},
                   goto = "__end__"
)


def code_gen(state: AgentState):
    sandbox.start()
    question = state["question"]
    msgs = [
        Generator_PROMPT,                
        HumanMessage(content=question) 
    ]
    code_soln = cast(GeneratorOutput, GENERATOR.invoke(msgs))

    if code_soln is None:
        return {"code":"", "system_error": "Generator didnt provide any code soln"}
    
    code_soln = code_soln.model_dump() 
    thinking = code_soln.get("thinking")
    code = code_soln.get("code")
    charts_exists = code_soln.get("charts_exists")
    generated_chart_names = code_soln.get("generated_chart_names")

    return {"code": code, "thinking": thinking, "attempts": 1, "charts_exists": charts_exists, "generated_chart_names": generated_chart_names}


def code_execute(state: AgentState):
    if state.get("system_error"):
        return {"system_error": state.get("system_error")}
    
    code = state.get("code", "")
    if not code:
         return {"system_error": "Code was empty."}
    
    code_resp = sandbox.process.code_run(code)
    if code_resp.exit_code == 0:
        if state.get("charts_exists", False):
            download_charts(state.get("generated_chart_names", []))
        return {"answer": code_resp.result}
    else:
        return {"agent_error": f"Error: Code execution failed {code_resp.exit_code} {code_resp.result}"}



def cmd_execute(state: AgentState):
    cmd = state.get("cmd", "")
    if not cmd:
        return {"system_error": "Cmd was empty."} 
    cmd_resp = sandbox.process.exec(cmd)

    if cmd_resp.exit_code == 0:
        return {"agent_error": None}
    else:
        return {"agent_error": f"Error: Cmd execution failed {cmd_resp.exit_code} {cmd_resp.result}"}


def should_continue(state: AgentState):
    if state.get("answer") or state.get("system_error"):
        return "summarizer"
    elif state.get("attempts", 0) < 4 and state.get("agent_error"):
        return "reflector"
    else:
        return "summarizer"


def summarizer(state: AgentState):
    answer = state.get("answer")  
    agent_error = state.get("agent_error")
    system_error = state.get("system_error")
    charts_exists = state.get("charts_exists", False)
    generated_chart_names = state.get("generated_chart_names", [])
    thinking = state.get("thinking", "No thinking provided.")
    code = state.get("code", "No code provided.")
    
    if answer:  
        content = f"Analysis results:\n{answer}"
        if charts_exists and generated_chart_names:
            content += f"\n\nVisualizations created: {', '.join(generated_chart_names)}"
        
        summary = SUMMARIZER.invoke([Summarizer_PROMPT, AIMessage(content=thinking), AIMessage(content=code), HumanMessage(content=content)])
        return {"messages": [summary]}
    
    elif system_error:
        return {"messages": [AIMessage(content=f"System Failure: {system_error}")]}
    
    elif agent_error:
        failure_msg = f"I failed to find a solution after maximum attempts. The last error encountered was: {agent_error}"
        return {"messages": [AIMessage(content=failure_msg)]}
    
    else:
        return {"messages": [AIMessage(content="No result generated.")]}



def reflection(state: AgentState):
    code = state.get("code", "No code provided.")
    agent_error = state.get("agent_error", "No error reported.")
    thinking = state.get("thinking", "No thinking provided.")


    if not code:
        human_input_str = f"The previous step failed to generate code. Error: {agent_error}. Please generate a solution from scratch using thinking: {thinking}."
    else:
        human_input_str = (
            "Review the following code and error to reflect on the needed fix:\n\n"
            f"--- Code ---\n{code}\n"
            f"--- Agent Error ---\n{agent_error}\n"
            f"--- Thinking process of model who wrote the code ---\n{thinking}\n"
        )
        
    reflection = cast(ReflectorOutput, REFLECTOR.invoke(
        [Reflector_PROMPT, HumanMessage(content=human_input_str)]
    ))
    if reflection is None:
        return {"system_error": "Reflector LLM failed to generate response. Aborting."}

    reflection = reflection.model_dump()  
    fix_type = reflection.get("fix_type")
    cmd = None
    new_code = None
    if fix_type == "ENVIRONMENT_FIX":
        cmd = reflection.get("cmd")
    else:
        # Default to CODE_FIX
        new_code = reflection.get("code")

    return {
        "reflection": reflection, 
        "attempts": state.get("attempts", 0) + 1, 
        "fix_type": fix_type, 
        "code": new_code if new_code else state.get("code"),
        "cmd": cmd
    }

def router(state: AgentState):
        if state.get("fix_type") == "ENVIRONMENT_FIX":
            return "cmd_execute"
        else:
            return "code_execute"


graph = StateGraph(AgentState)
graph.add_node("manager_cmd", manager_cmd)
graph.add_node("code_gen", code_gen)
graph.add_node("code_execute", code_execute)
graph.add_node("summarizer", summarizer)
graph.add_node("reflector", reflection)
graph.add_node("cmd_execute", cmd_execute)


graph.add_edge(START, "manager_cmd")
graph.add_edge("code_gen", "code_execute")
graph.add_conditional_edges("code_execute", should_continue, ["summarizer", "reflector"])
graph.add_conditional_edges("reflector", router, ["cmd_execute", "code_execute"])
graph.add_edge("cmd_execute", "code_execute")
graph.add_edge("summarizer", END)

app = graph.compile()