import ast
import json
import os
from typing import Callable

import dependencies
from pyinstrument import Profiler
from pyinstrument.renderers.console import ConsoleRenderer

from .ai.ai import Ai
from .browser.browser import Browser
from .calendar.calendar import Calendar
from .clipboard.clipboard import Clipboard
from .contacts.contacts import Contacts
from .display.display import Display
from .docs.docs import Docs
from .files.files import Files
from .keyboard.keyboard import Keyboard
from .mail.mail import Mail
from .mouse.mouse import Mouse
from .os.os import Os
from .skills.skills import Skills
from .sms.sms import SMS
from .terminal.terminal import Terminal
from .vision.vision import Vision


class Computer:
    def __init__(self, interpreter):
        self.interpreter = interpreter

        self.terminal = Terminal(self)

        self.offline = False
        self.verbose = False
        self.debug = False

        self.mouse = Mouse(self)
        self.keyboard = Keyboard(self)
        self.display = Display(self)
        self.clipboard = Clipboard(self)
        self.mail = Mail(self)
        self.sms = SMS(self)
        self.calendar = Calendar(self)
        self.contacts = Contacts(self)
        self.browser = Browser(self)
        self.os = Os(self)
        self.vision = Vision(self)
        self.skills = Skills(self)
        self.docs = Docs(self)
        self.ai = Ai(self)
        self.files = Files(self)

        self.emit_images = True
        self.api_base = "https://api.openinterpreter.com/v0"
        self.save_skills = True

        self.import_computer_api = False  # Defaults to false
        self._has_imported_computer_api = False  # Because we only want to do this once

        self.import_skills = False
        self._has_imported_skills = False
        self.max_output = (
            self.interpreter.max_output
        )  # Should mirror interpreter.max_output

        self.system_message = """

# THE COMPUTER API

A python `computer` module is ALREADY IMPORTED, and can be used for many tasks:

```python
computer.browser.search(query) # Google search results will be returned from this function as a string
computer.files.edit(path_to_file, original_text, replacement_text) # Edit a file
computer.calendar.create_event(title="Meeting", start_date=datetime.datetime.now(), end_date=datetime.datetime.now() + datetime.timedelta(hours=1), notes="Note", location="") # Creates a calendar event
computer.calendar.get_events(start_date=datetime.date.today(), end_date=None) # Get events between dates. If end_date is None, only gets events for start_date
computer.calendar.delete_event(event_title="Meeting", start_date=datetime.datetime) # Delete a specific event with a matching title and start date, you may need to get use get_events() to find the specific event object first
computer.contacts.get_phone_number("John Doe")
computer.contacts.get_email_address("John Doe")
computer.mail.send("john@email.com", "Meeting Reminder", "Reminder that our meeting is at 3pm today.", ["path/to/attachment.pdf", "path/to/attachment2.pdf"]) # Send an email with a optional attachments
computer.mail.get(4, unread=True) # Returns the [number] of unread emails, or all emails if False is passed
computer.mail.unread_count() # Returns the number of unread emails
computer.sms.send("555-123-4567", "Hello from the computer!") # Send a text message. MUST be a phone number, so use computer.contacts.get_phone_number frequently here
```

Do not import the computer module, or any of its sub-modules. They are already imported.

    """.strip()

    # Shortcut for computer.terminal.languages
    @property
    def languages(self):
        return self.terminal.languages

    @languages.setter
    def languages(self, value):
        self.terminal.languages = value

    def run(self, *args, **kwargs):
        """
        Shortcut for computer.terminal.run
        """
        return self.terminal.run(*args, **kwargs)

    def exec(self, code):
        """
        Shortcut for computer.terminal.run("shell", code)
        It has hallucinated this.
        """
        return self.terminal.run("shell", code)

    def stop(self):
        """
        Shortcut for computer.terminal.stop
        """
        return self.terminal.stop()

    def terminate(self):
        """
        Shortcut for computer.terminal.terminate
        """
        return self.terminal.terminate()

    def screenshot(self, *args, **kwargs):
        """
        Shortcut for computer.display.screenshot
        """
        return self.display.screenshot(*args, **kwargs)

    def view(self, *args, **kwargs):
        """
        Shortcut for computer.display.screenshot
        """
        return self.display.screenshot(*args, **kwargs)

    def to_dict(self):
        def json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except:
                return False

        return {k: v for k, v in self.__dict__.items() if json_serializable(v)}

    def load_dict(self, data_dict):
        for key, value in data_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    #############################################################################################
    #############################################################################################
    # Context Functions [PUBLIC FUNCTIONS]
    #############################################################################################
    #############################################################################################

    def trace(self, func: Callable, *args, **kwargs):
        """
        Profile the execution of a function with given arguments.

        Args:
            func (Callable): The function to be profiled.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            str: The profiling output.
        """
        profiler = Profiler()

        # Start profiling
        profiler.start()

        # Call the function to be profiled
        func(*args, **kwargs)
        # self.search_manager.get_function_dependencies(function_name)

        # Stop profiling
        session = profiler.stop()

        # Render the profile
        profile_renderer = ConsoleRenderer(unicode=True, color=True, show_all=True)
        profile_output = profile_renderer.render(session)

        return profile_output

    def map(self, path: str = ""):
        """
        parsed_files: list[str] = []
            # list of all files ending with .py, which are likely not test files
            # These are all ABSOLUTE paths.

        class_index: ClassIndexType = {}
            # for file name in the indexes, assume they are absolute path
            # class name -> [(file_name, line_range)]

        class_func_index: ClassFuncIndexType = {}
            # {class_name -> {func_name -> [(file_name, line_range)]}}
            # inner dict is a list, since we can have (1) overloading func names,
            # and (2) multiple classes with the same name, having the same method

        function_index: FuncIndexType = {}
            # function name -> [(file_name, line_range)]

        parent_index: ParentIndexType = {}
            # function name -> parent name (Class)

        dependencies_index: DependenciesIndexType = {}
            # function name -> [dependency names]
        """
        # handle empty path
        if not path:
            path = os.getcwd()

        # build indices
        (
            class_index,
            class_func_index,
            function_index,
            parsed_files,
            parent_index,
            dependencies_index,
        ) = dependencies._build_indices(path)

        # repository representation
        repository = dependencies._retrieve_repo(
            parsed_files,
            class_index,
            class_func_index,
            dependencies_index,
            parent_index,
            function_index,
        )

        return repository

    def search(self, project_path: str, query: str):
        # build indices
        (
            _,
            class_func_index,
            function_index,
            parsed_files,
            _,
            _,
        ) = dependencies._build_indices(project_path)

        # search code for query code
        tool_output, _, success = dependencies._search_code(
            project_path, parsed_files, query, class_func_index, function_index
        )

        if success:
            print(tool_output)

    def read(self, repo_path: str, file_path: str, line_no: int):
        """
        Given a file path and line number, extract the code snippet from the line number scope and display the
        parent structure of the function at scope line number.
        """
        # build indices
        (
            class_index,
            class_func_index,
            function_index,
            _,
            _,
            dependencies_index,
        ) = dependencies._build_indices(repo_path)

        (
            class_name,
            class_range,
            func_name,
            func_range,
        ) = dependencies._file_line_to_class_and_func_ranges(
            file_path,
            line_no,
            class_func_index,
            dependencies_index,
            function_index,
            class_index,
        )
        if not func_name:
            return "no function in the scope of the given line_no"
        if not func_range:
            return "no function range was found"

        start = 0
        if class_range:
            start = class_range[0]

        end = 0
        if func_range:
            end = func_range[0]

        # get the code snippet between func_name:start and class_name:start using search_manager.retrieve_code_snippet(line_nums=True)
        intermediary_code = dependencies._retrieve_code_snippet(
            file_path, start, end - 1, line_nums=False
        )

        # tree = ast.parse the code snippet
        intermediary_code_tree = ast.parse(intermediary_code)

        intermediary_parent_funcs = []
        # loop through ast.walk(tree)
        for node in ast.walk(intermediary_code_tree):
            if isinstance(node, ast.FunctionDef):
                # extract the function signature using search_utils.extract_func_sig_from_ast
                node_name = str(node.name)

                node_range = None
                result = function_index.get(node_name)
                if result:
                    _, node_range = result[0]

                if not node_range:
                    print("error no range for node: ", node_name)
                    return

                function_sig = dependencies._get_code_snippets_with_lineno(
                    file_path, node_range[0], node_range[0]
                )

                func_name_dependencies = dependencies_index.get(node.name)

                # if the function node has func_name as a dependency, add it to the output
                if func_name_dependencies:
                    for name in func_name_dependencies:
                        if func_name in name:
                            intermediary_parent_funcs.append(function_sig)

        output = ""

        # once we've finished walking the tree, prepend the class name and line to the output
        if class_name:
            result = class_index.get(class_name)
            if result:
                _, line_range = result[0]
                class_start = line_range.start
                class_sig = f"{class_start}: class {class_name}:\n"
                output += class_sig
                output += "\n ... \n"

        for function in intermediary_parent_funcs:
            output += function
            output += "\n ... \n"

        # append the function implementation to the output
        function_impl = dependencies._retrieve_code_snippet(
            file_path, func_range[0], func_range[1], line_nums=True
        )
        output += function_impl
        output += "\n ... \n"

        # return the output
        return output
