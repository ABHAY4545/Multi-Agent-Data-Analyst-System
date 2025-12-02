# from daytona import CreateSandboxFromSnapshotParams, CodeLanguage
# params = CreateSandboxFromSnapshotParams(name="JobStore", language=CodeLanguage.PYTHON)
# sandbox = daytona.create(params=params)

from daytona import Daytona
from schema import GeneratorOutput, ReflectorOutput, ManagerOutput
from pydantic import SecretStr
import os
from langchain_openai import ChatOpenAI

sandbox = Daytona().find_one("CodeStore")

def execute_code(code):
    resp = sandbox.process.code_run(code)
    if resp.exit_code != 0:
        return {"error": f"Error: Code execution failed {resp.exit_code} {resp.result}"}
    else:
        return {"answer": resp.result}
    
def execute_cmd(cmd):
    resp = sandbox.process.exec(cmd)
    if resp.exit_code == 0:
        return resp.result
    else:
        return {"error": f"Error: Cmd execution failed {resp.exit_code} {resp.result}"}
    
openrouter_config = {
    "model": "inception/mercury-coder",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
}

lm_config = {
    "model": "openai/gpt-oss-20b",
    "base_url": "http://127.0.0.1:1234/v1", 
    "api_key": SecretStr("OPENAI_API_KEY"),
    "model_kwargs":{"response_format": {"type": "json_object"}}
}
def create_models(config: dict):
    MANAGER = ChatOpenAI(**config).with_structured_output(ManagerOutput)
    GENERATOR = ChatOpenAI(**config).with_structured_output(GeneratorOutput)
    REFLECTOR = ChatOpenAI(**config).with_structured_output(ReflectorOutput)
    SUMMARIZER = ChatOpenAI(**config)
    return MANAGER, GENERATOR, REFLECTOR, SUMMARIZER


def download_charts(chart_names: list[str]):
    for chart_name in chart_names:
        sandbox.fs.download_file(chart_name, chart_name)
        print(f"Downloaded {chart_name}")