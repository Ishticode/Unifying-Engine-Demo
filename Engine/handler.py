import importlib
import Engine

framework_stack = []


def choose_framework(framework: str):
    global framework_stack
    backend = importlib.import_module("Engine.frameworks." + framework)
    framework_stack.append(backend)
    for key, value in backend.__dict__.items():
        Engine.__dict__[key] = value

