#!/usr/bin/env python

import importlib
import pkgutil

from typer import Typer

import sci_soft_models

###############################################################################

app = Typer()

###############################################################################

IGNORE_MODULES = [
    "bin",
    "types",
    "constants",
    "main",
    "data",
]


@app.command()
def run_command(
    model: str,
    task: str,
    use_coiled: bool = False,
    coiled_vm_type: str = "",
    coiled_min_workers: int = -1,
    coiled_max_workers: int = -1,
    coiled_keepalive: str = "",
) -> None:
    """Run a command."""
    # Normalize the model name
    model = model.lower().replace("-", "_")

    # Check that model is in the list of available models
    sci_soft_models_modules = [
        mod
        for mod in pkgutil.iter_modules(sci_soft_models.__path__)
        if mod.name not in IGNORE_MODULES
    ]

    # Ensure it is a valid model
    current_models = [mod.name for mod in sci_soft_models_modules]
    if model not in current_models:
        raise ValueError(
            f"Model {model} not found in current list of models: {current_models}."
        )

    # Import the model
    model_module = importlib.import_module(f"sci_soft_models.{model}")

    # Check that task is in the list of available tasks
    model_task_modules = [
        mod
        for mod in pkgutil.iter_modules(model_module.__path__)
        if mod.name not in IGNORE_MODULES
    ]

    # Ensure it is a valid task
    current_tasks = [mod.name for mod in model_task_modules]
    if task not in current_tasks:
        raise ValueError(
            f"Task {task} not found in current list of tasks: {current_tasks}."
        )

    # Import the task
    task_module = importlib.import_module(f"sci_soft_models.{model}.{task}")

    # Handle coiled kwargs
    coiled_kwargs: dict[str, int | str] = {}
    for key, value in {
        "coiled_vm_type": coiled_vm_type,
        "coiled_min_workers": coiled_min_workers,
        "coiled_max_workers": coiled_max_workers,
        "coiled_keepalive": coiled_keepalive,
    }.items():
        if isinstance(value, str) and value != "":
            coiled_kwargs[key] = value
        elif isinstance(value, int) and value > 0:
            coiled_kwargs[key] = value

    # Run the task
    print(f"Running sci_soft_models.{model}.{task}...")
    task_module.run(
        use_coiled=use_coiled,
        **coiled_kwargs,
    )


###############################################################################


def main() -> None:
    app()


if __name__ == "__main__":
    main()
