import importlib
import pkgutil
import inspect

def get_modules(path):
    modules = {}

    for loader, name, is_pkg in pkgutil.walk_packages(path):
        module = loader.find_module(name).load_module(name)
        for name, value in inspect.getmembers(module):
            # Only import classes we defined
            if not inspect.isclass(value) or value.__module__ is not module.__name__:
                continue
            modules[name] = value

    return modules

def __classname_to_modulename(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def __str_to_class(module_name, class_name):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            logging.error('Class does not exist')
    except ImportError:
        logging.error('Module does not exist')
    return class_ or None
