import ast
import builtins
import glob
import pathlib
import re
import textwrap
from collections import defaultdict, namedtuple
from collections.abc import MutableMapping
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path

#############################################################################################
#############################################################################################
# Type and Constant Declarations
#############################################################################################
#############################################################################################

RESULT_SHOW_LIMIT = 3

LineRange = namedtuple("LineRange", ["start", "end"])
ClassIndexType = MutableMapping[str, list[tuple[str, LineRange]]]
ClassFuncIndexType = MutableMapping[
    str, MutableMapping[str, list[tuple[str, LineRange]]]
]
FuncIndexType = MutableMapping[str, list[tuple[str, LineRange]]]
ParentIndexType = MutableMapping[str, str]
DependenciesIndexType = MutableMapping[str, set[str]]


@dataclass
class SearchResult:
    """Dataclass to hold search results."""

    file_path: str  # this is absolute path
    class_name: str | None
    func_name: str | None
    code: str

    def to_tagged_upto_file(self, project_root: str):
        """Convert the search result to a tagged string, upto file path."""
        rel_path = _to_relative_path(self.file_path, project_root)
        file_part = f"<file>{rel_path}</file>"
        return file_part

    def to_tagged_upto_class(self, project_root: str):
        """Convert the search result to a tagged string, upto class."""
        prefix = self.to_tagged_upto_file(project_root)
        class_part = (
            f"<class>{self.class_name}</class>" if self.class_name is not None else ""
        )
        return f"{prefix}\n{class_part}"

    def to_tagged_upto_func(self, project_root: str):
        """Convert the search result to a tagged string, upto function."""
        prefix = self.to_tagged_upto_class(project_root)
        func_part = (
            f" <func>{self.func_name}</func>" if self.func_name is not None else ""
        )
        return f"{prefix}{func_part}"

    def to_tagged_str(self, project_root: str):
        """Convert the search result to a tagged string."""
        prefix = self.to_tagged_upto_func(project_root)
        code_part = f"<code>\n{self.code}\n</code>"
        return f"{prefix}\n{code_part}"

    @staticmethod
    def collapse_to_file_level(lst, project_root: str) -> str:
        """Collapse search results to file level."""
        res = dict()  # file -> count
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = 1
            else:
                res[r.file_path] += 1
        res_str = ""
        for file_path, count in res.items():
            rel_path = _to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            res_str += f"- {file_part} ({count} matches)\n"
        return res_str

    @staticmethod
    def collapse_to_method_level(lst, project_root: str) -> str:
        """Collapse search results to method level."""
        res = dict()  # file -> dict(method -> count)
        for r in lst:
            if r.file_path not in res:
                res[r.file_path] = dict()
            func_str = r.func_name if r.func_name is not None else "Not in a function"
            if func_str not in res[r.file_path]:
                res[r.file_path][func_str] = 1
            else:
                res[r.file_path][func_str] += 1
        res_str = ""
        for file_path, funcs in res.items():
            rel_path = _to_relative_path(file_path, project_root)
            file_part = f"<file>{rel_path}</file>"
            for func, count in funcs.items():
                if func == "Not in a function":
                    func_part = func
                else:
                    func_part = f" <func>{func}</func>"
                res_str += f"- {file_part}{func_part} ({count} matches)\n"
        return res_str


#############################################################################################
#############################################################################################
# map() Dependencies [PRIVATE FUNCTIONS]
#############################################################################################
#############################################################################################


def _retrieve_repo(
    parsed_files: list[str],
    class_index: ClassIndexType,
    class_func_index: ClassFuncIndexType,
    dependencies_index: DependenciesIndexType,
    parent_index: ParentIndexType,
    function_index: FuncIndexType,
) -> dict:
    """
    Returns a nested hashmap representation of the repository structure, including function dependencies.
    """
    repo_context = {}
    class_methods = []

    for file_path in parsed_files:
        file_context = {"type": "file", "classes": {}, "functions": {}}

        # Add classes and their methods
        for class_name, class_info in class_index.items():
            for file_name, (start, end) in class_info:
                if file_name == file_path:
                    class_context = {
                        "type": "class",
                        "start_line": start,
                        "end_line": end,
                        "methods": {},
                    }
                    if class_name in class_func_index:
                        for method_name, method_info in class_func_index[
                            class_name
                        ].items():
                            class_methods.append(method_name)
                            for method_file, (method_start, method_end) in method_info:
                                if method_file == file_path:
                                    class_context["methods"][method_name] = {
                                        "type": "method",
                                        "start_line": method_start,
                                        "end_line": method_end,
                                        "dependencies": dependencies_index.get(
                                            method_name, []
                                        ),
                                        "parent": parent_index.get(method_name, None),
                                    }
                    file_context["classes"][class_name] = class_context

        # Add top-level functions
        for func_name, func_info in function_index.items():
            if func_name in class_methods:
                continue

            for func_file, (start, end) in func_info:
                if func_file == file_path:
                    file_context["functions"][func_name] = {
                        "type": "function",
                        "start_line": start,
                        "end_line": end,
                        "dependencies": dependencies_index.get(func_name, []),
                        "parent": parent_index.get(func_name, None),
                    }

        repo_context[file_path] = file_context

    return repo_context


def _build_indices(
    project_path: str,
) -> tuple[
    ClassIndexType,
    ClassFuncIndexType,
    FuncIndexType,
    list[str],
    ParentIndexType,
    DependenciesIndexType,
]:
    class_index: ClassIndexType = defaultdict(list)
    class_func_index: ClassFuncIndexType = defaultdict(lambda: defaultdict(list))
    function_index: FuncIndexType = defaultdict(list)
    parent_index: ParentIndexType = defaultdict()
    dependencies_index: DependenciesIndexType = defaultdict(set)

    py_files = _find_python_files(project_path)
    # holds the parsable subset of all py files
    parsed_py_files = []
    for py_file in py_files:
        file_info = _parse_python_file(py_file)
        if file_info is None:
            # parsing of this file failed
            continue
        parsed_py_files.append(py_file)
        # extract from file info, and form search index
        classes, class_to_funcs, top_level_funcs = file_info

        # (1) build class index
        for c, start, end in classes:
            class_index[c].append((py_file, LineRange(start, end)))
            parent_index[c] = "top"  # top-level classes have no parent

        # (2) build class-function index
        for c, class_funcs in class_to_funcs.items():
            for f, start, end in class_funcs:
                class_func_index[c][f].append((py_file, LineRange(start, end)))
                parent_index[f] = c  # function's parent is the class

        # (3) build (top-level) function index
        for f, start, end in top_level_funcs:
            function_index[f].append((py_file, LineRange(start, end)))
            if (
                f not in parent_index
            ):  # check that the function is not in the parent_index already
                parent_index[f] = "top"  # top-level function have no parent

        # Extract dependencies
        for c, class_funcs in class_to_funcs.items():
            for f, start, end in class_funcs:
                dependencies = _extract_dependencies(py_file, start, end)
                dependencies_index[f].update(dependencies)

        for f, start, end in top_level_funcs:
            dependencies = _extract_dependencies(py_file, start, end)
            dependencies_index[f].update(dependencies)

    return (
        class_index,
        class_func_index,
        function_index,
        parsed_py_files,
        parent_index,
        dependencies_index,
    )


def _find_python_files(dir_path: str) -> list[str]:
    """Get all .py files recursively from a directory.

    Skips files that are obviously not from the source code, such third-party library code.

    Args:
        dir_path (str): Path to the directory.
    Returns:
        List[str]: List of .py file paths. These paths are ABSOLUTE path!
    """

    py_files = glob.glob(pjoin(dir_path, "**/*.py"), recursive=True)
    res = []
    for file in py_files:
        rel_path = file[len(dir_path) + 1 :]
        if rel_path.startswith("build"):
            continue
        if rel_path.startswith("doc"):
            # discovered this issue in 'pytest-dev__pytest'
            continue
        if rel_path.startswith("requests/packages"):
            # to walkaround issue in 'psf__requests'
            continue
        if (
            rel_path.startswith("tests/regrtest_data")
            or rel_path.startswith("tests/input")
            or rel_path.startswith("tests/functional")
        ):
            # to walkaround issue in 'pylint-dev__pylint'
            continue
        if rel_path.startswith("tests/roots") or rel_path.startswith(
            "sphinx/templates/latex"
        ):
            # to walkaround issue in 'sphinx-doc__sphinx'
            continue
        if rel_path.startswith("tests/test_runner_apps/tagged/") or rel_path.startswith(
            "django/conf/app_template/"
        ):
            # to walkaround issue in 'django__django'
            continue
        res.append(file)
    return res


def _parse_python_file(file_full_path: str) -> tuple[list, dict, list] | None:
    """
    Main method to parse AST and build search index.
    Handles complication where python ast module cannot parse a file.
    """
    try:
        file_content = pathlib.Path(file_full_path).read_text()
        tree = ast.parse(file_content)
    except Exception:
        # failed to read/parse one file, we should ignore it
        return None

    # (1) get all classes defined in the file
    classes = []
    # (2) for each class in the file, get all functions defined in the class.
    class_to_funcs = {}
    # (3) get top-level functions in the file (exclues functions defined in classes)
    top_level_funcs = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            ## class part (1): collect class info
            class_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno
            # line numbers are 1-based
            classes.append((class_name, start_lineno, end_lineno))

            ## class part (2): collect function info inside this class
            class_funcs = [
                (n.name, n.lineno, n.end_lineno)
                for n in ast.walk(node)
                if isinstance(n, ast.FunctionDef)
            ]
            class_to_funcs[class_name] = class_funcs

        elif isinstance(
            node, ast.FunctionDef
        ):  # does this parse all functions or only top-level ones?
            function_name = node.name
            start_lineno = node.lineno
            end_lineno = node.end_lineno
            # line numbers are 1-based
            top_level_funcs.append((function_name, start_lineno, end_lineno))

    return classes, class_to_funcs, top_level_funcs


def _extract_dependencies(file_full_path: str, start: int, end: int) -> list[str]:
    """Extract dependencies from a code snippet in the given range."""
    # Get a set of built-in function names
    builtin_functions = set(dir(builtins))

    keywords_to_filter = {
        "if",
        "elif",
        "while",
        "for",
        "with",
        "match",
        "except",
        "assert",
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "any",
        "all",
    }

    with open(file_full_path) as f:
        file_content = f.readlines()

    code_snippet = "".join(file_content[start - 1 : end])
    code_snippet = textwrap.dedent(code_snippet)

    return _parse_with_ast(code_snippet, builtin_functions, keywords_to_filter)


def _parse_with_ast(code, builtin_functions, keywords_to_filter):
    tree = ast.parse(code)

    class DependencyVisitor(ast.NodeVisitor):
        def __init__(self):
            self.dependencies = set()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                full_name = self.get_full_attribute_name(node.func)
                if not any(full_name.startswith(kw) for kw in keywords_to_filter):
                    self.dependencies.add(full_name)
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                if (
                    func_name not in builtin_functions
                    and func_name not in keywords_to_filter
                ):
                    self.dependencies.add(func_name)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            full_name = self.get_full_attribute_name(node)
            if not any(full_name.startswith(kw) for kw in keywords_to_filter):
                self.dependencies.add(full_name)
            self.generic_visit(node)

        def get_full_attribute_name(self, node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))

    visitor = DependencyVisitor()
    visitor.visit(tree)
    return list(visitor.dependencies)


#############################################################################################
#############################################################################################
# search() Dependencies [PRIVATE FUNCTIONS]
#############################################################################################
#############################################################################################


def _search_code(
    project_path: str,
    parsed_files: list[str],
    code_str: str,
    class_func_index: ClassFuncIndexType,
    function_index: FuncIndexType,
) -> tuple[str, str, bool]:
    # attempt to search for this code string in all py files
    all_search_results: list[SearchResult] = []
    for file_path in parsed_files:
        searched_line_and_code: list[
            tuple[int, str]
        ] = _get_code_region_containing_code(file_path, code_str)
        if not searched_line_and_code:
            continue
        for searched in searched_line_and_code:
            line_no, code_region = searched
            # from line_no, check which function and class we are in
            class_name, func_name = _file_line_to_class_and_func(
                file_path, line_no, class_func_index, function_index
            )
            code_with_lineno = _get_code_snippets_with_lineno(
                file_path, line_no, line_no + len(code_region.split("\n"))
            )
            res = SearchResult(file_path, class_name, func_name, code_with_lineno)
            all_search_results.append(res)

    if not all_search_results:
        tool_output = f"Could not find code {code_str} in the codebase."
        summary = tool_output
        return tool_output, summary, False

    # good path
    tool_output = f"Found {len(all_search_results)} snippets containing `{code_str}` in the codebase:\n\n"
    summary = tool_output

    if len(all_search_results) > RESULT_SHOW_LIMIT:
        tool_output += "They appeared in the following files:\n"
        tool_output += SearchResult.collapse_to_file_level(
            all_search_results, project_path
        )
    else:
        for idx, res in enumerate(all_search_results):
            res_str = res.to_tagged_str(project_path)
            tool_output += f"- Search result {idx + 1}:\n```\n{res_str}\n```\n"
    return tool_output, summary, True


def _get_code_region_containing_code(
    file_full_path: str, code_str: str
) -> list[tuple[int, str]]:
    """In a file, get the region of code that contains a specific string.

    Args:
        - file_full_path: Path to the file. (absolute path)
        - code_str: The string that the function should contain.
    Returns:
        - A list of tuple, each of them is a pair of (line_no, code_snippet).
        line_no is the starting line of the matched code; code snippet is the
        source code of the searched region.
    """
    with open(file_full_path) as f:
        file_content = f.read()

    context_size = 3
    # since the code_str may contain multiple lines, let's not split the source file.

    # we want a few lines before and after the matched string. Since the matched string
    # can also contain new lines, this is a bit trickier.
    pattern = re.compile(re.escape(code_str))
    # each occurrence is a tuple of (line_no, code_snippet) (1-based line number)
    occurrences: list[tuple[int, str]] = []
    for match in pattern.finditer(file_content):
        matched_start_pos = match.start()
        # first, find the line number of the matched start position (1-based)
        matched_line_no = file_content.count("\n", 0, matched_start_pos) + 1
        # next, get a few surrounding lines as context
        search_start = match.start() - 1
        search_end = match.end() + 1
        # from the matched position, go left to find 5 new lines.
        for _ in range(context_size):
            # find the \n to the left
            left_newline = file_content.rfind("\n", 0, search_start)
            if left_newline == -1:
                # no more new line to the left
                search_start = 0
                break
            else:
                search_start = left_newline
        # go right to fine 5 new lines
        for _ in range(context_size):
            right_newline = file_content.find("\n", search_end + 1)
            if right_newline == -1:
                # no more new line to the right
                search_end = len(file_content)
                break
            else:
                search_end = right_newline

        start = max(0, search_start)
        end = min(len(file_content), search_end)
        context = file_content[start:end]
        occurrences.append((matched_line_no, context))

    return occurrences


def _file_line_to_class_and_func(
    file_path: str,
    line_no: int,
    class_func_index: ClassFuncIndexType,
    function_index: FuncIndexType,
) -> tuple[str | None, str | None]:
    """
    Given a file path and a line number, return the class and function name.
    If the line is not inside a class or function, return None.
    """
    # check whether this line is inside a class
    for class_name in class_func_index:
        func_dict = class_func_index[class_name]
        for func_name, func_info in func_dict.items():
            for file_name, (start, end) in func_info:
                if file_name == file_path and start <= line_no <= end:
                    return class_name, func_name

    # not in any class; check whether this line is inside a top-level function
    for func_name in function_index:
        for file_name, (start, end) in function_index[func_name]:
            if file_name == file_path and start <= line_no <= end:
                return None, func_name

    # this file-line is not recorded in any of the indexes
    return None, None


def _get_code_snippets_with_lineno(file_full_path: str, start: int, end: int) -> str:
    """Get the code snippet in the range in the file.

    The code snippet should come with line number at the beginning for each line.

    TODO: When there are too many lines, return only parts of the output.
          For class, this should only involve the signatures.
          For functions, maybe do slicing with dependency analysis?

    Args:
        file_path (str): Path to the file.
        start (int): Start line number. (1-based)
        end (int): End line number. (1-based)
    """
    with open(file_full_path) as f:
        file_content = f.readlines()

    snippet = ""
    for i in range(start - 1, end):
        snippet += f"{i+1} {file_content[i]}"
    return snippet


#############################################################################################
#############################################################################################
# read() Dependencies [PRIVATE FUNCTIONS]
#############################################################################################
#############################################################################################


def _file_line_to_class_and_func_ranges(
    file_path: str,
    line_no: int,
    class_func_index: ClassFuncIndexType,
    dependencies_index: DependenciesIndexType,
    function_index: FuncIndexType,
    class_index: ClassIndexType,
) -> tuple[str | None, tuple[int, int] | None, str | None, tuple[int, int] | None]:
    """
    Given a file path and a line number, return the class and function name.
    If the line is not inside a class or function, return None.
    """
    # check whether this line is inside a class
    for class_name in class_func_index:
        func_dict = class_func_index[class_name]
        for func_name, func_info in func_dict.items():
            for file_name, (start, end) in func_info:
                if file_name == file_path and start <= line_no <= end:
                    # we should check here in the dependencies of the file_name if any of the dependencies are in the top_level functions dictionary

                    func_dependencies = dependencies_index.get(func_name)

                    # if the function has no dependencies then we are in the right scope, return
                    while func_dependencies:
                        # otherwise, we might not be in the lowest scope and should exhaustively search
                        found_lower_scope = False

                        for dependency in func_dependencies:
                            if dependency in function_index:
                                output = function_index[dependency]
                                _, line_range = output[0]

                                if line_range.start <= line_no <= line_range.end:
                                    # there is a lower level scope that is closer to line_no
                                    func_name = dependency
                                    (start, end) = (line_range.start, line_range.end)
                                    func_dependencies = dependencies_index.get(
                                        dependency
                                    )
                                    found_lower_scope = True
                                    break

                        # exit if we are in the lowest scope
                        if not found_lower_scope:
                            break

                    result = class_index.get(class_name)
                    if result:
                        _, line_range = result[0]
                        class_range = (line_range.start, line_range.end)
                    else:
                        class_range = None
                    return class_name, class_range, func_name, (start, end)

    # not in any class; check whether this line is inside a top-level function
    for func_name in function_index:
        for file_name, (start, end) in function_index[func_name]:
            if file_name == file_path and start <= line_no <= end:
                return None, None, func_name, (start, end)

    # this file-line is not recorded in any of the indexes
    return None, None, None, None


def _retrieve_code_snippet(
    file_path: str, start_line: int, end_line: int, line_nums: bool
) -> str:
    if line_nums:
        return _get_code_snippets_with_lineno(file_path, start_line, end_line)
    return _get_code_snippets(file_path, start_line, end_line)


def _get_code_snippets(file_full_path: str, start: int, end: int) -> str:
    """Get the code snippet in the range in the file, without line numbers.

    Args:
        file_path (str): Full path to the file.
        start (int): Start line number. (1-based)
        end (int): End line number. (1-based)
    """
    with open(file_full_path) as f:
        file_content = f.readlines()
    snippet = ""
    for i in range(start - 1, end):
        snippet += file_content[i]
    return snippet


#############################################################################################
#############################################################################################
# Util Functions [PRIVATE FUNCTIONS]
#############################################################################################
#############################################################################################


def _to_relative_path(file_path: str, project_root: str) -> str:
    """Convert an absolute path to a path relative to the project root.

    Args:
        - file_path (str): The absolute path.
        - project_root (str): Absolute path of the project root dir.

    Returns:
        The relative path.
    """
    if Path(file_path).is_absolute():
        return str(Path(file_path).relative_to(project_root))
    else:
        return file_path
