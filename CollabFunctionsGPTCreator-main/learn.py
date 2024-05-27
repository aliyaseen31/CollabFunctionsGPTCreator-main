import pprint
import subprocess
import traceback
import openai
import json
from typing import Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.llm_utils import UnifiedVectorDB, HumanLLMMonitor, load_prompt, save_prompt, _visual_input, is_vscode_installed, smart_print, smart_input
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from config import *
import os
import uuid
import re
import shutil
import hashlib

from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain_caching.db"))

openai.api_key = OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

UnifiedVectorDB.db_type = "elasticsearch"  # "elasticsearch" "chroma"
UnifiedVectorDB.es_url = elastic_url_port


class Environment:
    def __init__(self, temp_root_dir: str = None, data_dir: str = "data"):
        self.temp_root_dir = temp_root_dir if temp_root_dir else os.path.join(os.getcwd(), "temp")
        self.data_dir = data_dir
        self.current_temp_dir = None
        # create "backups" directory where saved states will be stored
        if not os.path.exists(os.path.join(self.temp_root_dir, "backups")):
            os.makedirs(os.path.join(self.temp_root_dir, "backups"))

    def reset(self, backup_previous_temp_dir=True):
        if self.current_temp_dir is not None:
            self.close(backup_previous_temp_dir)
        # create a new temp directory in temp_root_dir named with a uuid
        self.current_temp_dir = os.path.join(self.temp_root_dir, str(uuid.uuid4()))
        os.makedirs(self.current_temp_dir)
        # create a write-only link to the data directory in the temp directory
        os.symlink(os.path.abspath(self.data_dir), os.path.join(self.current_temp_dir, "data"), target_is_directory=True)

    def step(self, action_code, context={}):
        # memorize current directory, to allow to change to temp directory, then change back to memorized directory
        current_dir = os.getcwd()
        os.chdir(self.current_temp_dir)
        # Regular expression to check if the last line assigns to 'result'
        if not re.search(r'\bresult\s*=', action_code.strip().splitlines()[-1]):
            helper = "\nresult = locals().get('_', None)"
        else:
            helper = ""

        # execute action
        try:
            # capture stdout and stderr while executing code
            exec(action_code + helper, context)
            exec_result = context.get('result', [])
            no_runtime_error = True
        except Exception as e:
            exec_result = f"Failed to execute provided code. Error: {e} Traceback: {traceback.format_exc()}"
            no_runtime_error = False
        # set execution environment back to the memorized directory
        os.chdir(current_dir)
        return no_runtime_error, exec_result

    def close(self, backup_previous_temp_dir=True):
        # move temp directory and its content including the data link to backups directory
        if backup_previous_temp_dir:
            shutil.move(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups"))
        else:
            shutil.rmtree(self.current_temp_dir)

    def backup_state(self, unique_id: str = None):
        # copy all the temp directory (excluding data directory) into a folder named by unique_id into backups directory
        if unique_id is None:
            unique_id = str(uuid.uuid4())
        shutil.copytree(self.current_temp_dir, os.path.join(self.temp_root_dir, "backups", unique_id), ignore=shutil.ignore_patterns('data'))
        return unique_id

    def restore_state(self, unique_id):
        # copy all the content of the backup directory into the temp directory (excluding data directory)
        self.reset(backup_previous_temp_dir=False)
        # copy all the content of the backup directory into the temp directory which already contains the data directory
        shutil.copytree(os.path.join(self.temp_root_dir, "backups", unique_id), self.current_temp_dir, ignore=shutil.ignore_patterns('data'), dirs_exist_ok=True)

    def get_state(self, extended: bool = False):
        # return a dictionary containing the content of the temp directory
        state = {}
        for root, dirs, files in os.walk(self.current_temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    file_content = f.read()
                state[file_path] = hashlib.sha256(file_content).hexdigest()
        # convert the dictionary into a string useful for comparison and analysis by language models
        state_text = "Files directory content: " + (json.dumps(state) if state.keys().__len__() > 0 else "empty")
        return state_text

    def get_score(self):
        # return the score of the current state
        return None


class EnvironmentManager:
    def __init__(self, env_type="default"):
        if env_type == "techsynthesis":
            from env.IR_CPS_TechSynthesis.env import VoyagerEnvIR_CPS_TechSynthesis
            self.env = VoyagerEnvIR_CPS_TechSynthesis()
        else:
            self.env = Environment()
        self.env.reset()

    def get_environment(self):
        return self.env


# Agent 1: Task Identification
class TaskIdentificationAgent:
    def __init__(self, llm, envs: [Environment], premium_llm=None, problem_prompts_subdir=None):
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.llm = llm
        self.premium_llm = premium_llm
        self.learnt_tasks: Dict[str, str] = {}
        self.failed_tasks: Dict[str, str] = {}
        self.human_llm_identify_best_task = HumanLLMMonitor(llm=self.llm, premium_llm=self.premium_llm, premium_llm_by_default=True)
        self.envs = envs

    def update_learnt_tasks(self, tasks: Dict[str, str]) -> None:
        self.learnt_tasks = tasks

    def update_failed_tasks(self, tasks: Dict[str, str]) -> None:
        self.failed_tasks = tasks

    def identify_best_task(self) -> str:
        learnt_tasks = format(json.dumps(self.learnt_tasks))
        failed_tasks = format(json.dumps(self.failed_tasks))
        envs_status = '\n'.join([env.get_state() for env in self.envs])
        user_message = f"- Already developed tasks: {learnt_tasks if learnt_tasks and learnt_tasks != '{}' else 'None'}\n" + \
                       f"- Already failed tasks (too hard): {failed_tasks if failed_tasks and failed_tasks != '{}' else 'None'}\n" + \
                       f"- Current status of examples on which the task will be tested on: {envs_status}\n"
        task = self.human_llm_identify_best_task.CallHumanLLM(system_prompt_template=self.problem_prompts_subdir + "identify_best_task", user_message=user_message, return_message_content_only=False)
        return task


# Agent 2: Code Task
class CodingAgent:
    def __init__(self, llm, envs: [Environment], premium_llm=None, problem_prompts_subdir=None, db_collection_success="successful_tasks", db_collection_failed="failed_tasks"):
        # super().__init__(llm)
        self.name = self.__class__.__name__
        self.problem_prompts_subdir = "" if problem_prompts_subdir is None else problem_prompts_subdir + "/"
        self.llm = llm
        self.premium_llm = premium_llm
        self.human_llm_code_task = HumanLLMMonitor(llm=self.llm, premium_llm=self.premium_llm, premium_llm_by_default=True, num_parallel_inferences=4)
        self.envs = envs
        self.db_successful_tasks = UnifiedVectorDB(collection_name=db_collection_success, embedding_function=HumanLLMMonitor.common_vectordb_embedding_function, persist_directory=HumanLLMMonitor.common_vectordb_persist_directory + db_collection_success)
        self.db_failed_tasks = UnifiedVectorDB(collection_name=db_collection_failed, embedding_function=HumanLLMMonitor.common_vectordb_embedding_function, persist_directory=HumanLLMMonitor.common_vectordb_persist_directory + db_collection_failed)

    def process_ai_generated_code(self, message, language="py", retry=3, required_bot_arg=None, task_definition=None):
        import ast
        import time
        import re
        error = None
        while retry > 0:
            try:
                if language == "py":  # Python case
                    # Match Python code blocks
                    code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
                    code = "\n".join(code_pattern.findall(message))

                    tests_pattern = re.compile(r'\n#\s+[Dd]ocument #([a-z0-9-]+)\s+usage test[^\n]*\n([^\n]+)')
                    tests = tests_pattern.findall(message)
                    # search also into the task definition
                    if task_definition is not None:
                        tests += tests_pattern.findall(task_definition)

                    parsed = ast.parse(code)
                    functions = []
                    imports = []

                    if len(code) == 0 or len(list(parsed.body)) == 0:
                        return False, f"Error parsing action response (No Code found): {parsed.body}"

                    main_function = None
                    runnable_code = ""
                    for node in parsed.body:
                        if isinstance(node, ast.FunctionDef):
                            node_type = "FunctionDef"
                            main_function = {
                                "name": node.name,
                                "type": node_type,
                                "body": ast.get_source_segment(code, node),
                                "params": [arg.arg for arg in node.args.args],
                            }
                            functions.append(main_function)
                        elif isinstance(node, ast.Expr) or isinstance(node, ast.Expression) or isinstance(node, ast.Assign):
                            node_type = "Expression"
                            runnable_code += "\n" + ast.get_source_segment(code, node)
                        elif isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                            node_type = "ImportFrom"
                            imports.append(ast.get_source_segment(code, node))
                            smart_print("ImportFrom node: IMPORT SHOULD BE DONE INSIDE FUNCTIONS !!!", self.name, "process_ai_generated_code SystemMessage")
                        else:
                            raise ValueError(f"Unsupported node type: {type(node)} - content:  {ast.get_source_segment(code, node)}")  # TODO: check if await is needed

                    assert main_function is not None, "No main function found."
                    if required_bot_arg:
                        assert required_bot_arg in main_function["params"], f"Main function {main_function['name']} must take an argument named '{required_bot_arg}'"

                    program_code = "\n".join(imports) + "\n"
                    program_code += "\n\n".join(function["body"] for function in functions)

                    for doc_id, test in tests:
                        try:
                            parsed_test = ast.parse(test)
                        except Exception as e:
                            return False, f"Error parsing code of Tests:\nERROR: {e}\nCODE: {test}"
                        # check if the test is a function call
                        if not isinstance(parsed_test.body[0], ast.Expr):
                            return False, f"Error parsing code of Tests (not a function call): {test}"
                else:
                    raise ValueError(f"Unsupported language in this version: {language}")

                return True, {
                    "program_code": program_code,
                    "main_function_name": main_function["name"],
                    "runnable_code": runnable_code,
                    "tests": tests,
                }

            except Exception as e:
                retry -= 1
                error = e
                time.sleep(0.1)

        return False, f"Error parsing action response (before program execution): {error}"

    def get_primitives(self):
        primitives = []
        # add to imports python text content of files located in the primitives directory which is located in the subdirectory of this file
        for root, dirs, files in os.walk(os.path.join(os.path.dirname(__file__), "primitives")):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    primitives.append(f.read())
        return primitives

    def code_task_and_run_test(self, refined_task: str, previous_errors=None, previous_scores=None, previous_codes=None, reset_unique_ids: str = None) -> str:
        primitives = self.get_primitives()
        user_message = f"TASK DEFINITION: {refined_task}"
        user_message += f"\n\nCURRENT STATE OF THE ENVIRONMENT USED TO TEST TASK:\n" + '\n'.join([env.get_state(extended=True) for env in self.envs])
        if len(primitives) > 0:
            user_message += f"\n\nCODE PRIMITIVES RE-USABLE OR FOR DEMONTRATION PURPOSE:\n" + '\n'.join(primitives)
        # TODO: include best code from successful tasks from CapitalizationAgent self.db_successful_tasks and self.db_failed_tasks
        smart_print("\033[91mTODO: add support to include best code from successful tasks from CapitalizationAgent self.db_successful_tasks and self.db_failed_tasks\033[0m", self.name, "code_task_and_run_test SystemMessage")
        if previous_errors:
            user_message += f"\n\nPREVIOUS ATTEMPTS TO CODE THE TASK: [[[\n"
            for previous_error, previous_score, previous_code in zip(previous_errors, previous_scores, previous_codes):
                user_message += f"\n\n<<\nERROR OR FEEDBACK: {previous_error.content}" + (f"\n\nSCORE: {previous_score}" if previous_score else "") + f"\n\nCODE: {previous_code}\n>>"
            user_message += f"\n]]]"
        codes = self.human_llm_code_task.CallHumanLLM(system_prompt_template=self.problem_prompts_subdir + "code_task", user_message=user_message, return_message_content_only=False, stream_output=False)
        results = []
        if reset_unique_ids is None:
            reset_unique_ids = [env.backup_state() for env in self.envs]
        # else:
        #     [env.restore_state(reset_unique_ids[id]) for id, env in enumerate(self.envs)]
        processed_codes = set()
        for code in codes:
            try:
                code_parsing_success, parsed_code = self.process_ai_generated_code(code.content, task_definition=refined_task)
                if parsed_code["program_code"] in processed_codes:
                    continue  # Skip the current iteration if this program code has already been processed to avoid duplicates
                else:
                    processed_codes.add(parsed_code["program_code"])
                smart_print(f"************ Code parsed result************\n{parsed_code}\n************************".replace("\\n", "\n"), self.name, "code_task_and_run_test RESULT")
                if code_parsing_success:
                    # Set initial state before running tests or runnable code
                    [env.restore_state(reset_unique_ids[id]) for id, env in enumerate(self.envs)]
                    # Initialize variables for runtime errors and execution results
                    no_runtime_errors, exec_results = [], []
                    # insert content of config.py into the code to ensure that the OPENAI_API_KEY is set
                    with open("config.py", "r") as f:
                        common_code = f.read() + "\n"
                    # Common code part to be executed in all cases
                    common_code += "\n".join(primitives) + "\n"
                    # Run the code in each environment
                    for env in self.envs:
                        # Determine tests to run or set default runnable code
                        matching_tests = [test for doc_id, test in parsed_code["tests"] if doc_id == env.id] if parsed_code["tests"] else [parsed_code['runnable_code']]
                        if not matching_tests:
                            no_runtime_error, exec_result = False, f"Error: no test found for given id {env.id}" if parsed_code["tests"] else f"Error: no runnable code found nor tests"
                        else:
                            # Concatenate common code with program and tests or runnable code
                            code_to_run = common_code + parsed_code["program_code"] + "\n" + "\n".join(matching_tests)
                            no_runtime_error, exec_result = env.step(code_to_run)
                            while not no_runtime_error and self.human_llm_code_task.skip_rounds <= 0:
                                smart_print("\033[31mCODE ERROR\033[0m: " + exec_result, self.name, "code_task_and_run_test SystemMessage")
                                if (input("Do you want to edit the code to fix the error (it may occur for each test of this code)? (yes/no): ").strip().lower() not in ("yes", "y")):
                                    break
                                edited_code = _visual_input(parsed_code["program_code"], filetype="py")
                                code_to_run = common_code + edited_code + "\n" + "\n".join(matching_tests)
                                smart_print("\033[31mTESTING NEW CODE\033[0m", self.name, "code_task_and_run_test SystemMessage")
                                no_runtime_error, exec_result = env.step(code_to_run)
                                # Update parsed_code if re-run is successful
                                if no_runtime_error:
                                    parsed_code["program_code"] = edited_code
                        # Append the results for each environment
                        no_runtime_errors.append(no_runtime_error)
                        exec_results.append(exec_result)
                    # Return combined results
                    results.append((parsed_code, all(no_runtime_errors), exec_results, reset_unique_ids, [env.get_score() for env in self.envs], [env.get_state(extended=True) for env in self.envs]))
                else:
                    results.append((code, False, parsed_code, None, None, None))
            except Exception as e:
                print(f"Skipping 1 code attempt - Error: {e} Traceback: {traceback.format_exc()}")
        if len(results) > 1:
            results_list = ""
            for id, result in enumerate(results):
                if result[1]:
                    results_list += f"{id}. SUCCESS / SCORE: {result[4]} / CODE: {result[0]['program_code'][:100]}\n"
                else:
                    results_list += f"{id}. \033[31mFAILED\033[0m / SCORE: {result[4]} / EXCEPTION: {result[2][0][:100]} / CODE: {result[0]['program_code'][:100]}\n"

            selected_code = input(f"{results_list}CODE SELECTION Please select the code to keep (separated by comma, or just hit enter to keep ALL): ").strip().replace(" ", "").lower().split(",")
            results = [result for id, result in enumerate(results) if selected_code and (str(id) in selected_code or selected_code == [""])]
        return results


# Agent 3: Code Validation
class ValidationAgent:
    def __init__(self, llm, envs: [Environment], premium_llm=None):
        # super().__init__(llm)
        self.name = self.__class__.__name__
        self.llm = llm
        self.premium_llm = premium_llm
        self.human_llm_validate_code = HumanLLMMonitor(llm=self.llm, premium_llm=self.premium_llm, premium_llm_by_default=False)
        self.envs = envs

    def validate_code(self, code: str, no_runtime_error: bool, exec_result: str, task: str = None, human_evaluation_required=False, scores=None, env_states=None) -> str:
        runtime_errors = f'\033[32mno runtime errors at execution - code returned:\n{exec_result}\n\033[0m' if no_runtime_error else f'\033[31mruntime errors at execution - error:{exec_result}\033[0m'
        if human_evaluation_required:
            human_evaluation = input(f"\n\n*******************\n{code}\n************\nCODE ABOVE EXECUTED with result: {runtime_errors}\n****\System may not efficiently evaluate what is produced by the code, please add your evaluation of the result (or hit enter): ")
        else:
            human_evaluation = ""
        runtime_errors = 'no runtime errors at execution' if no_runtime_error else 'runtime errors at execution'
        envs_status = '\n'.join(env_states)

        user_message = f"Task: {task}\n\n" + \
                       f"Code: {code}\n\n" + \
                       f"Code execution returned: {runtime_errors}\n\n" + \
                       f"Execution result returned by exec command of code provided: {exec_result}\n\n" + \
                       (f"Human evaluation of the result: {human_evaluation}\n\n" if human_evaluation != "" else "") + \
                       f"Performance scores: {scores}\n" + \
                       f"New environment status of examples on which the task has been tested on: {envs_status}\n"

        code_validation = self.human_llm_validate_code.CallHumanLLM(system_prompt_template="validate_code", user_message=user_message, return_message_content_only=False)
        return code_validation


# Agent 4: Code Capitalization
class CapitalizationAgent:
    def __init__(self, llm, db_collection_success="successful_tasks", db_collection_failed="failed_tasks", db_embedding_function=None, db_perist_directory=None, premium_llm=None):
        self.name = self.__class__.__name__
        self.tasks_repository: Dict[str, str] = {}
        self.failed_tasks_repository: Dict[str, str] = {}
        self.llm = llm
        self.premium_llm = premium_llm
        self.human_llm_generate_function_description = HumanLLMMonitor(llm=self.llm, premium_llm=self.premium_llm, premium_llm_by_default=False)
        self.db_successful_tasks = UnifiedVectorDB(
            collection_name=db_collection_success,
            embedding_function=db_embedding_function if db_embedding_function else HumanLLMMonitor.common_vectordb_embedding_function,
            persist_directory=db_perist_directory if db_perist_directory else HumanLLMMonitor.common_vectordb_persist_directory + db_collection_success)
        self.db_failed_tasks = UnifiedVectorDB(
            collection_name=db_collection_failed,
            embedding_function=db_embedding_function if db_embedding_function else HumanLLMMonitor.common_vectordb_embedding_function,
            persist_directory=db_perist_directory if db_perist_directory else HumanLLMMonitor.common_vectordb_persist_directory + db_collection_failed)

    def capitalize_successful_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket
        import uuid
        import datetime

        tool_description = str(self.generate_tool_description(parsed_code["main_function_name"], parsed_code["program_code"]))
        self.tasks_repository[parsed_code["main_function_name"]] = [tool_description, parsed_code["program_code"]]
        smart_print(f"************ Last added task ************\n{parsed_code['main_function_name']}\n************************".replace("\\n", "\n"), self.name, "capitalize_successful_tasks SUCCESS")

        function_file_path = os.path.join("functions", parsed_code["main_function_name"] + ".py")
        if os.path.exists(function_file_path):
            smart_print(f"Function file {function_file_path} already exists, please provide a new name for the function.", self.name, "capitalize_successful_tasks WARNING")
            function_file_path = os.path.join("functions", input("New function name: ") + ".py")
        with open(function_file_path, "w") as function_file:
            docstring_pattern = re.compile(r'(""".*?""")', re.DOTALL)
            docstring = docstring_pattern.findall(tool_description)[0]
            parsed_code["program_code"] = re.sub(r"(def " + parsed_code["main_function_name"] + "\(.*?\):)", r'\1\n    ' + docstring, parsed_code["program_code"], count=1)
            function_file.write(parsed_code["program_code"])

        if is_vscode_installed():
            smart_print("Please modify the file opened in vscode if necessary, and save it (Ctrl + W) when you are ok to continue", self.name, "capitalize_successful_tasks INSTRUCTIONS")
            subprocess.run(["code", "--wait", function_file_path])

        serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            "main_function_name": parsed_code["main_function_name"],
            "program_code": parsed_code["program_code"],
            "tool_description": tool_description,
            "task_description": task_description,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        tags = {"host": socket.gethostname() + "-" + str(uuid.getnode()), "step_id": HumanLLMMonitor.step_id, }

        self.db_successful_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def capitalize_failed_tasks(self, task_description: str, parsed_code: str) -> None:
        import socket
        import uuid
        import datetime

        main_function_name = _visual_input(parsed_code["main_function_name"] if parsed_code is not None and "main_function_name" in parsed_code else "replace this text with a descriptive name of the function")
        task_description_refined = _visual_input(task_description)
        self.failed_tasks_repository[main_function_name] = task_description_refined
        smart_print(f"************ Last added failed task ************\n{main_function_name}\n************************".replace("\\n", "\n"), self.name, "capitalize_failed_tasks CAPITALIZE FAIL")

        serialized_entry = json.dumps({
            "time": datetime.datetime.now().isoformat(),
            "main_function_name": main_function_name,
            "program_code": parsed_code["program_code"] if parsed_code is not None and "program_code" in parsed_code else None,
            "task_description": task_description,
            "task_description_refined": task_description_refined,
        }, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))

        tags = {"host": HumanLLMMonitor.get_host_id(), "step_id": HumanLLMMonitor.step_id, }

        self.db_failed_tasks.add_texts(texts=[serialized_entry], metadatas=[tags])

    def generate_tool_description(self, program_name, program_code):
        user_message = f"MAIN FUNCTION: `{program_name}`\n\nFULL CODE:\n{program_code}"
        tool_description = self.human_llm_generate_function_description.CallHumanLLM(system_prompt_template="generate_function_description", user_message=user_message, return_message_content_only=True)
        return tool_description

    def retrieve_saved_tasks_in_db(self, query="", max_db_results=20):
        smart_print(f"************ Retrieving successful tasks from database - LIST:", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        results_success_db = self.db_successful_tasks.query(query_text="*", k=max_db_results)
        id = 0
        for result in results_success_db:
            id += 1
            page_content = result.page_content
            task_data = json.loads(page_content)
            smart_print(f"{id}: function name:{task_data['main_function_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        include_code = input(f"CONFIG When adding the functions description in successful tasks, do you want to also include the code (it may overflow the maximum prompt length but can also guide generation) ? (yes/no): ").strip().lower() in ["yes", "y"]
        selected_functions = input(f"CONFIG Please select the functions to load (separated by comma, or 'all' to load all, or just hit enter for none): ").strip().replace(" ", "").lower().split(",")
        id = 0
        for result in results_success_db:
            id += 1
            if selected_functions and (str(id) not in selected_functions) and (selected_functions != ["all"]):
                continue
            page_content = result.page_content
            task_data = json.loads(page_content)
            if task_data["main_function_name"] in self.tasks_repository:
                smart_print(f"> function/task {task_data['main_function_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                continue
            self.tasks_repository[task_data["main_function_name"]] = [task_data["tool_description"]] if not include_code else [task_data["tool_description"], task_data["program_code"]]
            smart_print(f"> function/task {task_data['main_function_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")

        smart_print(f"************ Retrieving failed tasks from database - LIST:", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        results_failed_db = self.db_failed_tasks.query(query_text="*", k=max_db_results)
        id = 0
        for result in results_failed_db:
            id += 1
            page_content = result.page_content
            task_data = json.loads(page_content)
            smart_print(f"{id}: failed function name:{task_data['main_function_name']} time:{task_data['time']} host:{result.metadata['host']}", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
        selected_functions = input(f"CONFIG Please select the functions to load (separated by comma, or 'all' to load all, or just hit enter for none): ").strip().replace(" ", "").lower().split(",")
        id = 0
        for result in results_failed_db:
            id += 1
            if selected_functions and (str(id) not in selected_functions) and (selected_functions != ["all"]):
                continue
            page_content = result.page_content
            task_data = json.loads(page_content)
            if task_data["main_function_name"] in self.failed_tasks_repository:
                smart_print(f"> failed function/task {task_data['main_function_name']} already loaded. When there are duplicates select your prefered. Skipping...", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")
                continue
            self.failed_tasks_repository[task_data["main_function_name"]] = task_data["task_description_refined"]
            smart_print(f"> failed function/task {task_data['main_function_name']} from host {result.metadata['host']} generated at {task_data['time']} loaded.", self.name, "retrieve_saved_tasks_in_db DATABASE ACCESS")


def orchestrate_agents():
    # Initialize the Langchain llm
    default_llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    premium_llm = ChatOpenAI(model_name="gpt-4-1106-preview")

    problem_prompts_subdirs = [name for name in os.listdir("prompts") if os.path.isdir(os.path.join("prompts", name))]
    default_subdir = problem_prompts_subdirs[0] if problem_prompts_subdirs else ""
    choice = smart_input("CONFIG Enter a number for subdirectory (leave empty for default): " + "; ".join(f"{i}. {subdir}" for i, subdir in enumerate(problem_prompts_subdirs, 1)) + " ?", "CONFIG")
    problem_prompts_subdir = problem_prompts_subdirs[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(problem_prompts_subdirs) else default_subdir

    if "CPS" in problem_prompts_subdir:
        env_type = "techsynthesis"
        extra_manual_validation_to_capitalize = False
        documents = [{'id': "cf0d353c-b43b-4a79-88f9-42c2c84cf75e",
                      'title': "Complex QA and language models hybrid architectures, Survey",
                      'context': "This paper reviews the state-of-the-art of language models architectures and strategies for 'complex' question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and analyze limits and challenges of LLM in terms of tasks complexity and strict evaluation on accuracy (e.g. fairness, robustness, toxicity, ...) as a baseline. We discuss some challenges associated with complex QA, including domain adaptation, decomposition and efficient multi-step QA, long form and non-factoid QA, safety and multi-sensitivity data protection, multimodal search, hallucinations, explainability and truthfulness, temporal reasoning. We analyze current solutions and promising research trends, using elements such as: hybrid LLM architectural patterns, training and prompting strategies, active human reinforcement learning supervised with AI, neuro-symbolic and structured knowledge grounding, program synthesis, iterated decomposition and others.",
                      'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Complex QA and language models hybrid architectures Survey.json"},
                     {'id': "42252c6c-12f3-4edf-9045-8acd69bc3356",
                      'title': "Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature",
                      'context': "This paper surveys the empirical literature of inflation targeting. The main findings from our review are the following: there is robust empirical evidence that larger and more developed countries are more likely to adopt the IT regime; the introduction of this regime is conditional on previous disinflation, greater exchange rate flexibility, central bank independence, and higher level of financial development; the empirical evidence has failed to provide convincing evidence that IT itself may serve as an effective tool for stabilizing inflation expectations and for reducing inflation persistence; the empirical research focused on advanced economies has failed to provide convincing evidence on the beneficial effects of IT on inflation performance, while there is some evidence that the gains from the IT regime may have been more prevalent in the emerging market economies; there is not convincing evidence that IT is associated with either higher output growth or lower output variability; the empirical research suggests that IT may have differential effects on exchange-rate volatility in advanced economies versus EMEs; although the empirical evidence on the impact of IT on fiscal policy is quite limited, it supports the idea that IT indeed improves fiscal discipline; the empirical support to the proposition that IT is associated with lower disinflation costs seems to be rather weak. Therefore, the accumulated empirical literature implies that IT does not produce superior macroeconomic benefits in comparison with the alternative monetary strategies or, at most, they are quite modest.",
                      'target_file_path': "env/IR_CPS_TechSynthesis/document_embedding_analysis/output/arxiv/Macroeconomic Effects of Inflation Targeting A Survey of the Empirical  Literature.json"}]
        envs = []
        for doc in documents:
            env = EnvironmentManager(env_type).get_environment()
            env.title, env.abstract, env.synthesis_manager.target_file_path, env.id = doc['title'], doc['context'], doc['target_file_path'], doc['id']
            envs.append(env)

    else:
        env_type = "default"
        extra_manual_validation_to_capitalize = True
        manager = EnvironmentManager(env_type)
        envs = [manager.get_environment()]

    agent_taskreco = TaskIdentificationAgent(default_llm, envs, premium_llm=premium_llm, problem_prompts_subdir=problem_prompts_subdir)
    agent_coding = CodingAgent(default_llm, envs, premium_llm=premium_llm, problem_prompts_subdir=problem_prompts_subdir)
    agent_validation = ValidationAgent(default_llm, envs, premium_llm=premium_llm)
    agent_capitalize = CapitalizationAgent(default_llm, premium_llm=premium_llm)
    agent_capitalize.retrieve_saved_tasks_in_db()
    agent_taskreco.update_learnt_tasks(agent_capitalize.tasks_repository)
    agent_taskreco.update_failed_tasks(agent_capitalize.failed_tasks_repository)
    max_attempts = 4
    continue_identifying_tasks = True

    while continue_identifying_tasks:
        HumanLLMMonitor.step_id = str(uuid.uuid4())
        task = agent_taskreco.identify_best_task()
        if len(task) > 1:
            task_list = "Multiple task output, only one allowed - PLEASE SELECT:\n"
            for i, t in enumerate(task):
                task_list += f"\033[31m{i}\033[0m: {t.content[:200]}\n"
            smart_print(task_list, "orchestrate_agents", "orchestrate_agents SELECTION")
            while True:
                try:
                    task = task[int(input("Enter the index of the task to select: "))]
                    break
                except Exception as e:
                    print(f"Error: {e}\n\nEnter a valid index")
        else:
            task = task[0]
        smart_print("Identified Task: " + task.content.replace("\\n", "\n"), "orchestrate_agents", "orchestrate_agents RESULT")
        task_description = task.content
        parsed_code, validation = coding_and_validation_loop(agent_coding, agent_validation, task_description, max_attempts, extra_manual_validation_to_capitalize)
        if validation == "success":
            agent_capitalize.capitalize_successful_tasks(task_description, parsed_code)
            agent_taskreco.update_learnt_tasks(agent_capitalize.tasks_repository)
            continue_identifying_tasks = smart_input("CONFIG Enter 'yes' or 'y' to keep identifying tasks, otherwise enter anything else to exit", "CONFIG") in ["yes", "y"]
        else:
            agent_capitalize.capitalize_failed_tasks(task_description, parsed_code)
            agent_taskreco.update_failed_tasks(agent_capitalize.failed_tasks_repository)


def coding_and_validation_loop(agent_coding, agent_validation, task_description, max_attempts, extra_manual_validation_to_capitalize):
    task_description_refined = task_description
    previous_errors = []
    previous_scores = []
    previous_codes = []
    validation = "fail"

    for attempt in range(max_attempts):
        reset_unique_ids = None
        code_results = agent_coding.code_task_and_run_test(task_description_refined, previous_errors, previous_scores, previous_codes, reset_unique_ids)
        for code_result in code_results:
            code, no_runtime_errors, exec_results, reset_unique_ids, scores, env_states = code_result
            task_description_refined = task_description_refined if attempt == 0 else f"{task_description}\n<<\nPrevious attempt:\nTask refined with previous attempt comments: {previous_errors[attempt - 1]}\nTask refined with previous attempt code: {previous_codes[attempt - 1]}\n>>\nTask refined again\nTask refined with current attempt code: {code['program_code']}\n>>\nTask refined again\n<<"
            if no_runtime_errors:
                smart_print(f"\033[32mTASK CODE PARSED & EXECUTED SUCCESSFULLY:\n{code['program_code']}\nEXECUTION RESULT: {exec_results}\n\033[0m", "coding_and_validation_loop", "coding_and_validation_loop RESULT")
                # Print the result and check if manual validation is required before capitalizing
                if extra_manual_validation_to_capitalize:
                    if input(f"Do you want to capitalize the code ? (Enter 'yes' or 'y' to capitalize) ").strip().lower() in ["yes", "y"]:
                        validation = "success"
                    else:
                        validation = "fail"
                else:
                    validation = "success"
            else:
                previous_errors.append(exec_results)
                previous_codes.append(code)
                previous_scores.append(scores)

            if validation == "success":
                return code, validation

    return code, validation


if __name__ == "__main__":
    orchestrate_agents()
