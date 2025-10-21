"""
Module for the instantiate function and its private helper functions.
"""
import asyncio
import importlib
from pathlib import Path
from typing import Any, Optional, TypeVar


Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


def _get_public_functions_coroutines_and_classes(module_name: str) -> dict[str, Any]:
    """
    Import all public functions, coroutines, and classes from a specified module.

    Parameters:
        module_name (str): The target module's name.

    Returns:
        dict[str, Any]: A dictionary mapping attribute names to their corresponding 
                       objects. Keys are the names of public functions, coroutines, 
                       or classes, and values are the actual callable objects or 
                       coroutines.

    Example:
        >>> public_items = _get_public_functions_coroutines_and_classes('math')
        >>> 'sin' in public_items
        True
        >>> 'cos' in public_items
        True
    """
    module = importlib.import_module(module_name)
    return {
        attr_name: getattr(module, attr_name)
            for attr_name in dir(module)
        if (
            callable(getattr(module, attr_name)) or 
            asyncio.iscoroutine(getattr(module, attr_name))
        ) and not attr_name.startswith('_') # Ignore private functions
        and not isinstance(getattr(module, attr_name), type) # Ignore types
    }


def _get_resources_for(class_name: str, configs, name: str) -> dict[str, Any]:
    """
    Get resources (public functions, coroutines, and classes) 
    for a specific module from its resources folder.

    Args:
        class_name: Name of the class/module for which to retrieve resources.
        directory_chain: Optional[str|Path] = None
            Custom directory path to use instead of the default parent directory.
            If None, the parent directory of the current file is used.
    Returns:
        dict: A dictionary mapping function/class names to their callable objects.
    Notes:
        - Resources directory structure is expected to be:
          {leading_path}/{class_name}/resources/{class_name}
        - Only imports .py files that don't start with '_' or '__'
    """
    func_dict = {}

    # Get path to the class' resources folder.
    resources_folder = configs.paths.THIS_DIR / name / "resources" / class_name
    utils_folder = configs.paths.THIS_DIR / "utils"

    # See if we're importing a utility class.
    if not resources_folder.exists():
        utils_dirs = [dir.name for dir in utils_folder.iterdir() if dir.is_dir()]
        print(f"utils_dirs: {utils_dirs}")
        if class_name in utils_dirs:
            resources_folder = utils_folder / class_name / "resources"
        else:
            raise FileNotFoundError(f"Resources folder not found for {class_name} in {name} or utils")

    # Get all the functions in the class' resource folder.
    for file in resources_folder.iterdir():
        if not file.exists():
            continue
        if ( # Skip private and non-python files.
            file.is_file() 
            and file.suffix == ".py" 
            and not file.name.startswith("__") 
            and not file.name.startswith("_")
        ):
            # Load in the module, get the functions, and assign them to the dictionary.
            # Construct proper module name relative to the package
            package_parts = ["custom_nodes", "red_ribbon", name, "resources", class_name]
            module_name = f"{'.'.join(package_parts)}.{file.stem}"
            try:
                func_dict.update(
                    _get_public_functions_coroutines_and_classes(module_name)
                )
            except AssertionError as e:
                if "no Nodes have been registered" in str(e):
                    pass # Ignore reinitialization errors. Comfy will tell us that we haven't loaded anything anyways.
                else:
                    print(f"{type(e)} importing {module_name}: {e}")
            except Exception as e:
                print(f"{type(e)} importing {module_name}: {e}")

    # Print the functions that were loaded
    print(f"Loaded {len(func_dict)} functions for {class_name}:")
    for key, value in func_dict.items():
        print(f"key: {key}, value: {value}")
    return func_dict


def _make_class_instance(class_ : Class, class_name: str, configs, name: str) -> Any:
    """
    Instantiate a class with its resources and configurations.
    The class must have a constructor with only two values: resources(dict) and configs(Configs).

    Args:
        class_: The class to instantiate.
        class_name: The name of the class.
        configs: The configurations to pass to the class.
    Returns:
        Any: The instantiated class
    
    Notes:
        See: https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module
    """
    # cls = getattr(class_, class_name)
    return class_(_get_resources_for(class_name, configs, name), configs)


def instantiate(resources: dict[str, Class], configs, name: str) -> dict[str, ClassInstance]:
    """
    Instantiate all classes in the resources dictionary with their resources and configurations.

    Args:
        resources: A dictionary of classes to instantiate, with the class name as the key.
        configs: The configurations to pass to the classes.
        name: The name of the module to load resources from.

    Returns:
        dict[str, Any]: A dictionary of instantiated classes.
    """
    # Create a new dictionary to store instantiated classes
    instantiated = {}

    # Iterate over a copy of the resources dictionary to avoid modification during iteration
    for class_name, class_ in resources.copy().items():
        if class_ is not None:
            instantiated[class_name] = _make_class_instance(class_, class_name, configs, name)

    return instantiated