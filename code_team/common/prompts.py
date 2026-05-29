"""Shared prompt templates used by serial and codebase pipelines.

Moved from serial/prompts.py to decouple the codebase package from serial.
"""
from __future__ import annotations

import json

# Reuse tool-use instruction constants from the common package.
from common.prompt_constants import APPLY_PATCH_FORMAT_REQUIREMENTS, FILE_EDITOR_USAGE_REQUIREMENTS, GREP_GLOB_USAGE_REQUIREMENTS, TERMINAL_USAGE_REQUIREMENTS, INIT_PY_EXPORT_REQUIREMENTS, IMPORT_SOURCE_REQUIREMENTS


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


def architecture_prompt(
        requirement: str,
        uml_class: str,
        uml_sequence: str,
        arch_design: str,
        output_json_path: str,
        pre_existing_files: list[str] | None = None,
        feedback: str = "",
        history: str = "",
        integrity_rules: str = "",
) -> str:
    pre_existing_section = (
        "\n".join(f"- {f}" for f in pre_existing_files)
        if pre_existing_files else "(none)"
    )
    return f"""You are an expert software architect assistant.

You are given **four types of inputs** describing a software system:

Keep descriptions concise; avoid redundant explanations.

1. **PRD.md (Product Requirement Document)**
    - Contains natural language descriptions of functional and non-functional requirements.
    - May describe system components, data flow, user interactions, responsibilities.
2. **UML Class Diagram**
    - Describes classes and their relationships (inheritance, composition, usage).
    - Also indicates methods/functions defined in each class.
    - May include method signatures, including **parameter names and types**.
3. **UML Sequence Diagram**
    - Illustrates runtime interactions between objects/functions across different modules.
    - Specifies call orders, involved functions, and communicating components.
    - May also reveal **actual parameters used** during function calls.
4. **Architecture Design Document**
    - Lists the overall system structure (e.g., files, directories, and responsibilities).
    - May define which file belongs to which module, and what components a module encapsulates.

------

### **Your task:**

Based on the above inputs, you must extract and construct a **Repository Interface Blueprint (RIB)** — a structured representation of the system's architecture, containing modules, files, classes, and functions, along with their descriptions and relations.

Note: Global_functions in UML is a fake class used only to host global functions. Do not include it as a class in the RIB; instead, move its functions to the file-level "functions" list.

------

### **Target Output Format: Repository Interface Blueprint (RIB)**

RIB is a hierarchical, nested JSON tree with the following structure:

```
Module
 └── File
      ├── GlobalCode
      ├── Class
      │    └── Function
      └── Function
```

Each of these elements must contain the following fields:

- **`name`**: the identifier of the element, such as the module name, file name, class name, or function name (without path).

- **`description`**: brief natural language summary of its responsibility.

- **`path`** (only for File-level elements): the full relative path of the file from the project root directory (e.g., `"src/utils/io.py"`). This field reflects the actual file system structure, not the logical module hierarchy.

- **`parameters`** (only for Function-level elements): a list of parameter specifications, if available. Each parameter should be represented as a JSON object like:

  ```json
  {{"name": "param_name", "type": "param_type (if available)", "default": "default_value (if any)", "description": "brief description (if available)"}}
  ```
  - If a parameter has a default value mentioned anywhere in the documentation (PRD, README, UML, etc.), you **MUST** include a `"default"` field with the exact value.
  - If no default value is mentioned, omit the `"default"` field entirely.
- **`global_code`** (only for File-level elements): a list of top-level code snippets or statements not inside any class or function.

  This includes:
  - Definitions of global variables or constants (including enumerations defined at the file scope).
  - Executable initialization or setup logic that runs when the file is loaded.
  - Script entry point code outside any function.


------

### **Extraction Instructions:**

Please extract the necessary RIB components from each input as follows:

- **From PRD.md**:
  - Extract descriptions of high-level modules and their responsibilities
  - Identify any function- or class-level behavioral descriptions
- **From UML Class Diagram**:
  - Extract all classes and their methods
  - Capture class-to-class relationships (extends, uses, composition)
  - Add function declarations found in the class
  - When available, extract function parameters including names and types
- **From UML Sequence Diagram**:
  - Capture function call chains and interactions
  - Identify cross-module function relations (calls, communicates_with)
  - If parameter values are passed in interactions, infer parameter roles or examples
- **From Architecture Design Document**:
  - Extract mapping from files to modules
  - Capture file responsibilities and logical structure of the codebase
- If the UML Class Diagram or UML Sequence Diagram contains elements named Global_functions (or similar), treat them as a placeholder for global functions. Do not represent them as a class in the RIB. Instead, place the contained functions under the file's "functions" field (for global functions) or "global_code" field if they represent top-level executable code.

When extracting function parameters, if the documentation mentions default values (e.g., "default 265", "defaults to None", "optional, default is True"), you **MUST** include a `"default"` field in the parameter JSON. This is critical — missing defaults will cause downstream code generation failures.

------

### Output Example

Each module in the RIB is a JSON object like:

```
{{
  "name": "UserModule",
  "description": "Handles user-related operations",
  "files": [
    {{
      "name": "user_controller.py",
      "path": "src/controllers/user_controller.py",
      "description": "API handlers for user endpoints",
      "global_code": [
        {{
          "name": "USER_ROLES",
          "description": "List of valid user roles"
        }},
        {{
          "name": "init_logging",
          "description": "Sets up logging configuration at module load"
        }}
      ],
      "classes": [
        {{
          "name": "UserController",
          "description": "Coordinates user logic",
          "functions": [
            {{
              "name": "create_user",
              "description": "Creates a new user",
              "parameters": [
                {{"name": "user_data", "type": "dict", "description": "User input"}}
              ]
            }}
          ]
        }}
      ],
      "functions": [
        {{
          "name": "validate_username",
          "description": "Checks if the provided username is valid",
          "parameters": [
            {{"name": "username", "type": "str", "description": "The username to validate"}},
            {{"name": "max_length", "type": "int", "default": "50", "description": "Maximum allowed length for usernames"}}
          ]
        }}
      ]
    }}
  ]
}}
```

The final merged RIB is an array of such module objects: `[module1, module2, ...]`

------

### Tool-use Requirements

- Use `file_editor` for writing the RIB JSON.
{FILE_EDITOR_USAGE_REQUIREMENTS}
- For the output file, prefer `file_editor` with `command=create`.
- Do not use `apply_patch` for architecture JSON writes.
- After writing, validate JSON with `terminal`.
- File content at the end must be strict JSON only.
- Do not print the full RIB in final response.
- Final response must be strict JSON only:
{{"status":"ok","path":"{output_json_path}","note":"rib generated"}}

------

### **Inputs:**

#### **PRD.md**:
{requirement}

#### **UML Class Diagram**:
{uml_class} 

#### **UML Sequence Diagram**:
{uml_sequence}

#### **Architecture Design Document**:
{arch_design}

------

#### **Files Already Present in Workspace**

The following files already exist in the workspace as pre-populated data or resource
files (regardless of file extension). Do NOT create RIB entries for any of these
files — they are not source code to be generated. Even if a file has a `.py` extension,
if it appears in this list it is a data/resource file and must be excluded from the RIB.

{pre_existing_section}

------

{"" if not integrity_rules else integrity_rules + chr(10) + "------" + chr(10)}
Please generate the complete RIB JSON based on the given inputs.
Write the final result to: {output_json_path}

You should NOT put all files into a single module unless the input architecture explicitly states so.

Do NOT include test files or test directories (e.g. `unit_tests/`, `acceptance_tests/`, `tests/`, `test_*.py`) in the RIB. Tests are provided separately and should not be part of the architecture.

Only include source code files (`.py`) in the RIB. Do NOT include non-source files such as `LICENSE`, `README.md`, `.gitignore`, shell scripts (`.sh`), configuration files, or documentation — even if they appear in the architecture design file tree.

If any information is missing, make reasonable assumptions but clearly mark them in the descriptions using `"[assumed]"`.

Final response must be strict JSON only:
{{"status":"ok","path":"{output_json_path}","note":"rib generated"}}
"""


def architecture_prompt_dep(
        requirement: str,
        uml_class: str,
        uml_sequence: str,
        arch_design: str,
        output_json_path: str,
        pre_existing_files: list[str] | None = None,
        feedback: str = "",
        history: str = "",
        integrity_rules: str = "",
) -> str:
    pre_existing_section = (
        "\n".join(f"- {f}" for f in pre_existing_files)
        if pre_existing_files else "(none)"
    )
    return f"""You are an expert software architect assistant.

You are given **four types of inputs** describing a software system:

Keep descriptions concise; avoid redundant explanations.

1. **PRD.md (Product Requirement Document)**
    - Contains natural language descriptions of functional and non-functional requirements.
    - May describe system components, data flow, user interactions, responsibilities.
2. **UML Class Diagram**
    - Describes classes and their relationships (inheritance, composition, usage).
    - Also indicates methods/functions defined in each class.
    - May include method signatures, including **parameter names and types**.
3. **UML Sequence Diagram**
    - Illustrates runtime interactions between objects/functions across different modules.
    - Specifies call orders, involved functions, and communicating components.
    - May also reveal **actual parameters used** during function calls.
4. **Architecture Design Document**
    - Lists the overall system structure (e.g., files, directories, and responsibilities).
    - May define which file belongs to which module, and what components a module encapsulates.

------

### **Your task:**

Based on the above inputs, you must extract and construct a **Repository Interface Blueprint (RIB)** — a structured representation of the system's architecture, containing modules, files, classes, and functions, along with their descriptions and relations.

Note: Global_functions in UML is a fake class used only to host global functions. Do not include it as a class in the RIB; instead, move its functions to the file-level "functions" list.

------

### **Target Output Format: Repository Interface Blueprint (RIB)**

RIB is a hierarchical, nested JSON tree with the following structure:

```
Module
 └── File
      ├── GlobalCode
      ├── Class
      │    └── Function
      └── Function
```

Each of these elements must contain the following fields:

- **`name`**: the identifier of the element, such as the module name, file name, class name, or function name (without path).

- **`description`**: brief natural language summary of its responsibility.

- **`path`** (only for File-level elements): the full relative path of the file from the project root directory (e.g., `"src/utils/io.py"`). This field reflects the actual file system structure, not the logical module hierarchy.

- **`parameters`** (only for Function-level elements): a list of parameter specifications, if available. Each parameter should be represented as a JSON object like:

  ```json
  {{"name": "param_name", "type": "param_type (if available)", "default": "default_value (if any)", "description": "brief description (if available)"}}
  ```
  - If a parameter has a default value mentioned anywhere in the documentation (PRD, README, UML, etc.), you **MUST** include a `"default"` field with the exact value.
  - If no default value is mentioned, omit the `"default"` field entirely.
- **`global_code`** (only for File-level elements): a list of top-level code snippets or statements not inside any class or function.

  This includes:
  - Definitions of global variables or constants (including enumerations defined at the file scope).
  - Executable initialization or setup logic that runs when the file is loaded.
  - Script entry point code outside any function.

- **`dependencies`** (only for File-level elements): a JSON array of **internal file paths** that this file depends on. A file A depends on file B if A imports and uses any class, function, or constant **defined in B** — including usage in function bodies, parameter types, return types, and class inheritance. Rules:
  - Use the **exact `path` values** from other file entries in this RIB (e.g. `"bplustree/const.py"`, NOT module names like `"bplustree.const"`).
  - Do NOT include **transitive** dependencies: if A uses `Node` (from `node.py`) and `Node` internally uses `Entry` (from `entry.py`), then A depends on `node.py` but NOT `entry.py` — unless A also directly uses `Entry`.
  - Do NOT include standard library modules (os, sys, typing, etc.).
  - Do NOT include third-party packages (click, werkzeug, jinja2, etc.).
  - If a file has no internal dependencies, set `"dependencies": []`.
  - Every file entry MUST have this field.


------

### **Extraction Instructions:**

Please extract the necessary RIB components from each input as follows:

- **From PRD.md**:
  - Extract descriptions of high-level modules and their responsibilities
  - Identify any function- or class-level behavioral descriptions
- **From UML Class Diagram**:
  - Extract all classes and their methods
  - Capture class-to-class relationships (extends, uses, composition)
  - Add function declarations found in the class
  - When available, extract function parameters including names and types
- **From UML Sequence Diagram**:
  - Capture function call chains and interactions
  - Identify cross-module function relations (calls, communicates_with)
  - If parameter values are passed in interactions, infer parameter roles or examples
- **From Architecture Design Document**:
  - Extract mapping from files to modules
  - Capture file responsibilities and logical structure of the codebase
- If the UML Class Diagram or UML Sequence Diagram contains elements named Global_functions (or similar), treat them as a placeholder for global functions. Do not represent them as a class in the RIB. Instead, place the contained functions under the file's "functions" field (for global functions) or "global_code" field if they represent top-level executable code.

When extracting function parameters, if the documentation mentions default values (e.g., "default 265", "defaults to None", "optional, default is True"), you **MUST** include a `"default"` field in the parameter JSON. This is critical — missing defaults will cause downstream code generation failures.

------

### Output Example

Each module in the RIB is a JSON object like:

```
{{
  "name": "UserModule",
  "description": "Handles user-related operations",
  "files": [
    {{
      "name": "user_controller.py",
      "path": "src/controllers/user_controller.py",
      "dependencies": ["src/models/user.py", "src/utils/auth.py"],
      "description": "API handlers for user endpoints",
      "global_code": [
        {{
          "name": "USER_ROLES",
          "description": "List of valid user roles"
        }},
        {{
          "name": "init_logging",
          "description": "Sets up logging configuration at module load"
        }}
      ],
      "classes": [
        {{
          "name": "UserController",
          "description": "Coordinates user logic",
          "functions": [
            {{
              "name": "create_user",
              "description": "Creates a new user",
              "parameters": [
                {{"name": "user_data", "type": "dict", "description": "User input"}}
              ]
            }}
          ]
        }}
      ],
      "functions": [
        {{
          "name": "validate_username",
          "description": "Checks if the provided username is valid",
          "parameters": [
            {{"name": "username", "type": "str", "description": "The username to validate"}},
            {{"name": "max_length", "type": "int", "default": "50", "description": "Maximum allowed length for usernames"}}
          ]
        }}
      ]
    }},
    {{
      "name": "user.py",
      "path": "src/models/user.py",
      "dependencies": [],
      "description": "User data model",
      "global_code": [],
      "classes": [...],
      "functions": []
    }}
  ]
}}
```

The final merged RIB is an array of such module objects: `[module1, module2, ...]`

------

### Tool-use Requirements

- Use `file_editor` for writing the RIB JSON.
{FILE_EDITOR_USAGE_REQUIREMENTS}
- For the output file, prefer `file_editor` with `command=create`.
- Do not use `apply_patch` for architecture JSON writes.
- After writing, validate JSON with `terminal`.
- File content at the end must be strict JSON only.
- Do not print the full RIB in final response.
- Final response must be strict JSON only:
{{"status":"ok","path":"{output_json_path}","note":"rib generated"}}

------

### **Inputs:**

#### **PRD.md**:
{requirement}

#### **UML Class Diagram**:
{uml_class}

#### **UML Sequence Diagram**:
{uml_sequence}

#### **Architecture Design Document**:
{arch_design}

------

#### **Files Already Present in Workspace**

The following files already exist in the workspace as pre-populated data or resource
files (regardless of file extension). Do NOT create RIB entries for any of these
files — they are not source code to be generated. Even if a file has a `.py` extension,
if it appears in this list it is a data/resource file and must be excluded from the RIB.

{pre_existing_section}

------

{"" if not integrity_rules else integrity_rules + chr(10) + "------" + chr(10)}
Please generate the complete RIB JSON based on the given inputs.
Write the final result to: {output_json_path}

You should NOT put all files into a single module unless the input architecture explicitly states so.

Do NOT include test files or test directories (e.g. `unit_tests/`, `acceptance_tests/`, `tests/`, `test_*.py`) in the RIB. Tests are provided separately and should not be part of the architecture.

Only include source code files (`.py`) in the RIB. Do NOT include non-source files such as `LICENSE`, `README.md`, `.gitignore`, shell scripts (`.sh`), configuration files, or documentation — even if they appear in the architecture design file tree.

If any information is missing, make reasonable assumptions but clearly mark them in the descriptions using `"[assumed]"`.

Final response must be strict JSON only:
{{"status":"ok","path":"{output_json_path}","note":"rib generated"}}
"""


def architecture_judge_prompt(
    requirement: str,
    uml_class: str,
    uml_sequence: str,
    arch_design: str,
    architecture_json: str,
    pre_existing_files: list[str] | None = None,
) -> str:
    pre_existing_section = (
        "\n".join(f"- {f}" for f in pre_existing_files)
        if pre_existing_files else "(none)"
    )
    return f"""You are an expert software architecture reviewer.  

You will be given five inputs:  

1. **Requirements Document (PRD)** - describing the intended functionality of the system.  
2. **UML Class Diagram** - Describes classes and their relationships (inheritance, composition, usage).
3. **UML Sequence Diagram** - Illustrates runtime interactions between objects/functions across different modules.
4. **Architecture Design Document** - Lists the overall system structure (e.g., files, directories, and responsibilities).
5. **Proposed Architecture (ARCH)** - a generated architecture design that includes modules, components, and their dependencies.  

**Important Rule**:
If the PRD, UML diagrams, or Architecture Design Document explicitly provide a file directory structure, file names, or function names, you must prioritize consistency with this given information over general design principles. Do not suggest changes that contradict explicitly provided structures or names merely for abstract notions of modularity or cohesion. Your evaluation should respect and align with the provided information as the highest authority.
Do not suggest or generate any test files as part of the evaluation or revision.
When evaluating Consistency with Provided Information, ignore test files and test directories (e.g. `unit_tests/`, `acceptance_tests/`, `tests/`, `test_*.py`) — the ARCH is not expected to include them even if they appear in the reference documents.
Also ignore any files listed under "Files Already Present in Workspace" below — these are assets already present in the workspace (data files, config, templates, migrations, etc.) regardless of file extension, and do not need to be generated.

**PRD vs. UML/Architecture Naming**:
The PRD may use language-specific conventions (e.g., C-style API names) that differ from the actual class/function names defined in the UML diagrams and Architecture Design Document. When the PRD and UML/Architecture disagree on naming, **the UML and Architecture Design are authoritative** for naming; the PRD is authoritative only for functional requirements. Do not penalize the ARCH for using UML-defined names instead of PRD-style names. Only penalize if a core functional capability is genuinely missing.

Your task is to evaluate the quality of the proposed architecture (ARCH) based on the following four criteria:

1. **Requirement Coverage** - Does the architecture cover all the functional modules mentioned in the requirements? When the PRD uses language-specific API names, check for functional equivalents in the ARCH rather than exact name matches.
2. **Consistency with Provided Information** - Does the architecture faithfully follow the directory structure, file names, and function names explicitly given in the PRD, UML diagrams, and Architecture Design Document? **Focus on file completeness: every source file listed in the Architecture Design file tree must have a corresponding entry in the RIB. List any source files that appear in the Architecture Design file tree but are missing from the RIB (e.g. `__init__.py`). Each missing file is a deduction. Do not analyze other aspects under this criterion.**
3. **Interface Consistency** - Are the interface names clear, unambiguous, and free from redundancy?
4. **Dependency Relations** - Are there any circular dependencies? Does the dependency structure follow common layered architecture principles?  

For each criterion, provide a short justification of your evaluation.  
Then, give an **overall score** for the architecture between **1 (poor) and 10 (excellent)**, based on how well it satisfies the above aspects.  

Format your output as follows:  
```

Requirement Coverage: 
Consistency with Provided Information: 
Interface Consistency: 
Dependency Relations: 

Final Score: <a single number between 1 and 10>

```

------

### **Inputs:**

#### **PRD.md**:
{requirement}

#### **UML Class Diagram**:
{uml_class} 

#### **UML Sequence Diagram**:
{uml_sequence}

#### **Architecture Design Document**:
{arch_design}

#### **ARCH**:
{architecture_json}

------

#### **Files Already Present in Workspace**

{pre_existing_section}

""".strip()


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------

def _resolve_target_file_path(file_item: dict) -> str:
    target_path = ""
    file_block = file_item.get("file", {}) if isinstance(file_item, dict) else {}
    if isinstance(file_block, dict):
        target_path = str(file_block.get("path", "")).strip()
    if not target_path and isinstance(file_item, dict):
        target_path = str(file_item.get("path", "")).strip()
    return target_path


def rib_prompt(
    file_item: dict,
    architecture_summary: list[dict],
    feedback: str = "",
    history: str = "",
    integrity_rules: str = "",
) -> str:
    target_path = _resolve_target_file_path(file_item)
    return f"""You are an expert software project code generator.
    
Your task is to generate the **skeleton code** for a single file based on its RIB (Repository Interface Blueprint) description, while ensuring consistency with already generated files.

------

### ** Repository Interface Blueprint (RIB)**

RIB is a hierarchical, nested JSON tree with the following structure:

```
Module
 └── File
      ├── GlobalCode
      ├── Class
      │    └── Function
      └── Function
```

Each of these elements must contain the following fields:

- **`name`**: the identifier of the element, such as the module name, file name, class name, or function name (without path).

- **`description`**: brief natural language summary of its responsibility.

- **`path`** (only for File-level elements): the full relative path of the file from the project root directory (e.g., `"src/utils/io.py"`). This field reflects the actual file system structure, not the logical module hierarchy.

- **`parameters`** (only for Function-level elements): a list of parameter specifications, if available. Each parameter should be represented as a JSON object like:
  
  ```json
  {{"name": "param_name", "type": "param_type (if available)", "default": "default_value (if any)", "description": "brief description (if available)"}}
  ```
- **`global_code`** (only for File-level elements): a list of top-level code snippets or statements not inside any class or function.
        
  This includes:
  - Definitions of global variables or constants (including enumerations defined at the file scope).
  - Executable initialization or setup logic that runs when the file is loaded.
  - Script entry point code outside any function.

------

### Input Information

You will receive the following inputs:

1. **Project File Overview**
   - A summary of all files in the project (paths and descriptions).
   - If you need to see the code of any file for consistency (e.g. imports, shared types), use tools (`file_editor` view, `grep`, `glob`) to read it.
2. **Target File RIB Description**
   - `file`: the RIB entry describing this file (functions, classes, etc.)
   - `module`:
    - `name`: the module that this file belongs to
    - `description`: the purpose or role of this module in the project

### Output Information

- Generate **only the skeleton code** for the target file.
- Do **not** repeat the RIB description in the output.
- The skeleton code must include:
  - import statements
  - global variables, constants, or classes if present
  - function signatures (bodies replaced with `pass`)
- **Function signature rules**:
  - Follow the parameters listed in the RIB.
  - If a parameter has `"default": "None"`, write it as `=None` in the function signature.
  - If a parameter has another default value, use that exact default.
- If `"default"` is missing, leave the parameter without a default.
- Adds the function description as a comment immediately under each function signature.
- The file must be **syntactically valid Python code** and compilable.
- **Do not add any explanations or comments outside the code.**
{INIT_PY_EXPORT_REQUIREMENTS}
{IMPORT_SOURCE_REQUIREMENTS}


### Tool-use Requirements

- You must edit the project file directly using tools (prefer `apply_patch`).
- Target file path: `{target_path}`
{FILE_EDITOR_USAGE_REQUIREMENTS}
- You may use `glob` and `grep` to quickly locate symbols/usages in related files.
{GREP_GLOB_USAGE_REQUIREMENTS}
- If you need to check other files for import consistency, use `file_editor` to view them.
{APPLY_PATCH_FORMAT_REQUIREMENTS}
{TERMINAL_USAGE_REQUIREMENTS}
- Do not print the full skeleton code in final response.
- Final response must be strict JSON only:
{{"path":"{target_path}","status":"ok"}}

------

### Inputs:

#### 0. Feedback from Reviewer

{feedback}

#### 1. Relevant History

{history}

#### 2. Project File Overview

```json
{json.dumps(architecture_summary, ensure_ascii=False, indent=2)}
```

#### 3. Target File RIB Description

```json
{json.dumps(file_item, ensure_ascii=False, indent=2)}
```

------

{integrity_rules}
Please write skeleton code directly into the target file using tools.
Do not output code in final response. Output status JSON only.
""".strip()


def rib_judge_prompt(architecture: list[dict], skeleton: list[dict]) -> str:
    return f"""You are an expert software architecture reviewer.
You will be given two inputs:

1. **Architecture Specification (ARCH)** - the designed architecture, including the intended directory/file structure, module definitions, and interface specifications (classes, functions, parameters).
2. **Generated Skeleton (SKEL)** - the Python code skeleton produced from the architecture, including directory/file organization, imports, class definitions, and function signatures (with `pass` as placeholders).

Your task is to evaluate the quality of the generated skeleton based on the following two criteria:

1. **Directory Structure Matching** - Does the skeleton's directory and file hierarchy match the architecture specification? Are there missing or extra files/directories? Is the nesting consistent with the design?
2. **Interface & Call Relationship Matching** - Do the classes and functions (including names, parameters, and default values) align with the architecture definition? Are all expected interfaces present? Are there inconsistencies or omissions?

**Important Evaluation Guidelines**:
- The ARCH is the source of truth for the skeleton. If the ARCH intentionally adapts PRD conventions into a different paradigm (e.g., OOP instead of procedural C), the skeleton should match the ARCH, not the original PRD.
- Do NOT penalize the skeleton for design decisions already made at the architecture level. The skeleton judge evaluates ARCH-to-SKEL fidelity, not PRD-to-SKEL fidelity.
- Minor supportive additions in the skeleton (`__init__.py` files, import boilerplate, type hints) should not be penalized as "extra" elements.

For each criterion, provide a short justification of your evaluation.
 Then, give an **overall score** for the skeleton between **1 (poor) and 10 (excellent)**, based on how well it satisfies the above aspects.

Format your output as follows:

```
Directory Structure Matching:
Interface & Call Relationship Matching:

Final Score: <a single number between 1 and 10>
```

------

### **Inputs:**

#### **ARCH**:

```json
{json.dumps(architecture, ensure_ascii=False, indent=2)}
```

#### **SKEL**:

```json
{json.dumps(skeleton, ensure_ascii=False, indent=2)}
```


""".strip()

# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------

def _resolve_code_target_path(file_item: dict) -> str:
    if not isinstance(file_item, dict):
        return ""
    path = str(file_item.get("path", "")).strip()
    if path:
        return path
    file_block = file_item.get("file", {})
    if isinstance(file_block, dict):
        return str(file_block.get("path", "")).strip()
    return ""


def code_generation_prompt(
    file_item: dict,
    architecture_summary: list[dict],
    feedback: str = "",
    history: str = "",
    integrity_rules: str = "",
) -> str:
    target_path = _resolve_code_target_path(file_item)
    return f"""You are an expert software project code generator.

Your task is to generate the **full implementation code** for a single file, based on its provided skeleton and the project's architecture. 

### Requirements:
1. **Strict adherence to skeleton**  
   - Do not add new functions, classes, or methods that are not present in the skeleton.  
   - Do not remove or rename any existing functions, classes, or methods.  
   - Preserve the order and structure exactly as given.  

2. **Function implementation**  
   - Replace `pass` with the correct implementation according to the function name, parameters, and descriptions provided in the skeleton.  
   - Keep the function-level doc/comment (description) directly under the function signature.  
   - **Method call consistency:** Every method in the skeleton exists for a reason.
     Before implementing a method, check if other skeleton methods should be called
     within it. Prefer calling existing skeleton methods over reimplementing similar
     logic inline. You may add small private helpers for genuinely new logic, but do
     not use them to replace or duplicate what a skeleton method already provides.

   - **Architecture integrity:** Your implementation MUST genuinely use ALL classes,
     modules, and data structures defined in the skeleton. Do NOT bypass the designed
     architecture by substituting simpler alternatives (e.g., replacing a tree/graph
     structure with a plain dict or list, wrapping a built-in container to fake an
     interface). Each class must contain meaningful algorithmic logic as described in
     its docstrings. If the skeleton defines domain-specific classes (e.g., Node,
     Entry, Record), your code MUST instantiate and operate on them — not on
     surrogate containers.

3. **Consistency with context**  
   - Use the project file overview to understand sibling files.  
   - If you need to see other files for reference (imports, shared types, APIs), use tools (`file_editor` view, `grep`, `glob`) to read them.  
   - Do not modify other files.  

4. **Output format**  
   - Output only the complete code for the current file.  
   - Do not include any explanatory text outside of code.  

{INIT_PY_EXPORT_REQUIREMENTS}
{IMPORT_SOURCE_REQUIREMENTS}

### Tool-use Requirements

- You must edit the project file directly using tools (prefer `apply_patch`).
- Target file path: `{target_path}`
- If you need to check other files for consistency, use `file_editor` to view them.
{FILE_EDITOR_USAGE_REQUIREMENTS}
- You may use `glob` and `grep` for fast code search and symbol location.
{GREP_GLOB_USAGE_REQUIREMENTS}

{APPLY_PATCH_FORMAT_REQUIREMENTS}
{TERMINAL_USAGE_REQUIREMENTS}
- Do not print the full implementation code in final response.
- Final response must be strict JSON only:
{{"path":"{target_path}","status":"ok"}}

### Context Gathering (RECOMMENDED)

Before implementing, you SHOULD read example and data files in the workspace to
understand expected input/output behavior. This helps you write correct logic:
- Look for `examples/` directories containing sample inputs and expected outputs.
- Look for `data_file/` or similar directories with test datasets.
- For file-format converters (CSV→JSON, XML→dict, etc.), compare an input file
  with its corresponding output file to understand the exact transformation expected.
- These files are part of the project specification — reading them is encouraged.

### Boundaries (CRITICAL)

- Do NOT read, view, or access the `check_tests/` directory or any file inside it.
- Do NOT run `pytest` or any test command.
- You MAY run `python -m py_compile` for a quick syntax check, but this is optional
  since the main agent will run compile_check after you finish.
- Once you have written the complete implementation, finish immediately.
  Do not iterate on correctness — the main agent handles testing and fixes.

------

### Inputs:

#### Skeleton for the current file:

```json
{json.dumps(file_item, ensure_ascii=False, indent=2)}
```

#### Project File Overview:

```json
{json.dumps(architecture_summary, ensure_ascii=False, indent=2)}
```

#### Feedback

{feedback}

#### History

{history}

------

{integrity_rules}
Now, implement the target file by writing code directly into the target file using tools.
Do not output code in final response. Output status JSON only.
""".strip()


