import os
import typer
from rich import print
from rich.panel import Panel

from graphrag_toolkit.lexical_graph.prompts.prompt_provider_registry import PromptProviderRegistry
from graphrag_toolkit.lexical_graph.prompts.prompt_provider_factory import PromptProviderFactory

app = typer.Typer(help="Prompt Provider CLI for GraphRAG Toolkit")


def bootstrap_providers():
    """
    Register the default prompt provider (based on environment) under the name 'default'.
    You can extend this to register others explicitly.
    """
    default_provider = PromptProviderFactory.get_provider()
    PromptProviderRegistry.register("default", default_provider, default=True)


@app.command("env")
def show_env():
    """Show the current PROMPT_PROVIDER environment setting."""
    provider = os.getenv("PROMPT_PROVIDER", "static")
    print(f"[bold green]PROMPT_PROVIDER[/]: {provider}")


@app.command("list")
def list_providers():
    """List all registered prompt providers."""
    registered = PromptProviderRegistry.list_registered()
    if not registered:
        print("[yellow]No prompt providers registered.[/]")
    else:
        print("[bold underline]Registered Prompt Providers[/]")
        for name in registered:
            default = "(default)" if name == os.getenv("PROMPT_PROVIDER") else ""
            print(f"• [cyan]{name}[/] {default}")


@app.command("show")
def show_prompt(provider: str = typer.Option(..., help="Name of the registered provider")):
    """Show system and user prompts for a given provider."""
    instance = PromptProviderRegistry.get(provider)
    if not instance:
        print(f"[red]Provider not found:[/] {provider}")
        raise typer.Exit(code=1)

    try:
        system_prompt = instance.get_system_prompt()
        user_prompt = instance.get_user_prompt()

        print(Panel(str(system_prompt), title="System Prompt", expand=True))
        print(Panel(str(user_prompt), title="User Prompt", expand=True))

    except Exception as e:
        print(f"[red]Failed to load prompts:[/] {e}")
        raise typer.Exit(code=1) from e


@app.command("validate")
def validate_prompt(provider: str = typer.Option(..., help="Name of the provider to validate")):
    """Validate the prompt formats (text/JSON)."""
    instance = PromptProviderRegistry.get(provider)
    if not instance:
        print(f"[red]Provider not found:[/] {provider}")
        raise typer.Exit(code=1)

    try:
        for kind in ["system", "user"]:
            method = instance.get_system_prompt if kind == "system" else instance.get_user_prompt
            prompt = method()
            if isinstance(prompt, dict):
                print(f"[green]{kind.capitalize()} prompt is valid JSON[/]")
            elif isinstance(prompt, str):
                print(f"[blue]{kind.capitalize()} prompt is valid text[/]")
            else:
                print(f"[yellow]{kind.capitalize()} prompt returned unexpected type: {type(prompt)}[/]")
    except Exception as e:
        print(f"[red]Validation error:[/] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    bootstrap_providers()
    app()
