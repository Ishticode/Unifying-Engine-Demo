import importlib
import Engine

framework_stack = []
framework_stack_str = []


def choose_framework(framework: str):
    global framework_stack
    backend = importlib.import_module("Engine.frameworks." + framework)
    framework_stack.append(backend)
    framework_stack_str.append(framework)
    for key, value in framework_stack[-1].__dict__.items():
        Engine.__dict__[key] = value

