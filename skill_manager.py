"""
Skill Manager for WebArena agents.
Handles loading, parsing, and executing compound skills.
"""

import ast
import importlib.util
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("logger")


class SkillManager:
    """Manages compound skills for web automation."""

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Initialize the skill manager.

        Args:
            skills_dir: Path to directory containing skill files organized by site
        """
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.loaded_skills: Dict[
            str, Dict[str, Any]
        ] = {}  # site -> {skill_name -> function}
        self.skill_metadata: Dict[
            str, Dict[str, Dict]
        ] = {}  # site -> {skill_name -> metadata}

        if self.skills_dir and self.skills_dir.exists():
            self._load_all_skills()

    def _load_all_skills(self):
        """Load all skills from the skills directory."""
        if not self.skills_dir:
            return

        # Find all Python files in subdirectories
        for site_dir in self.skills_dir.iterdir():
            if site_dir.is_dir():
                site_name = site_dir.name
                if site_name.startswith("_"):
                    continue

                self.loaded_skills[site_name] = {}
                self.skill_metadata[site_name] = {}

                for skill_file in site_dir.glob("*.py"):
                    if skill_file.stem.startswith("_"):
                        continue  # Skip private files

                    try:
                        # Parse the file with AST to extract metadata (doesn't require imports)
                        with open(skill_file, "r") as f:
                            tree = ast.parse(f.read())

                        # Find all function definitions (both sync and async)
                        for node in ast.walk(tree):
                            if isinstance(
                                node, (ast.FunctionDef, ast.AsyncFunctionDef)
                            ) and not node.name.startswith("_"):
                                func_name = node.name

                                # Extract docstring
                                docstring = ast.get_docstring(node) or ""
                                description = (
                                    docstring.split("\n")[0]
                                    if docstring
                                    else f"Execute {func_name}"
                                )

                                # Extract parameters (skip 'page')
                                params = []
                                for arg in node.args.args:
                                    if arg.arg != "page":
                                        param_type = "Any"
                                        if arg.annotation:
                                            # Try to get the type annotation as string
                                            try:
                                                param_type = ast.unparse(
                                                    arg.annotation
                                                )
                                            except:
                                                param_type = "Any"

                                        params.append(
                                            {
                                                "name": arg.arg,
                                                "type": param_type,
                                                "default": None,  # AST parsing of defaults is complex, skip for now
                                            }
                                        )

                                # Store metadata
                                self.skill_metadata[site_name][func_name] = {
                                    "description": description,
                                    "parameters": params,
                                    "docstring": docstring,
                                    "file": str(skill_file),
                                }

                        # Now try to actually load the module for execution
                        try:
                            spec = importlib.util.spec_from_file_location(
                                f"skills.{site_name}.{skill_file.stem}",
                                skill_file,
                            )
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            # Store function references
                            for name, obj in inspect.getmembers(module):
                                if (
                                    inspect.isfunction(obj)
                                    or inspect.iscoroutinefunction(obj)
                                ) and not name.startswith("_"):
                                    self.loaded_skills[site_name][name] = obj
                                    logger.debug(
                                        f"Loaded function {name}: is_coroutine={inspect.iscoroutinefunction(obj)}"
                                    )

                            logger.info(
                                f"Loaded {len(self.loaded_skills[site_name])} functions from {skill_file}"
                            )

                        except Exception as e:
                            logger.warning(
                                f"Loaded metadata but cannot execute skills from {skill_file}: {e}"
                            )
                            # Still usable if we can at least show the descriptions

                    except Exception as e:
                        logger.warning(
                            f"Failed to parse skills from {skill_file}: {e}"
                        )

    def get_skills_for_sites(
        self, sites: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all skills available for the given sites.

        Args:
            sites: List of site names (e.g., ['shopping_admin', 'gitlab'])

        Returns:
            Dictionary mapping skill names to their functions
        """
        available_skills = {}
        for site in sites:
            if site in self.loaded_skills:
                available_skills.update(self.loaded_skills[site])
        return available_skills

    def get_skill_descriptions(self, sites: List[str]) -> str:
        """
        Generate a formatted description of available skills for prompting.

        Args:
            sites: List of site names

        Returns:
            Formatted string describing available skills
        """
        if not self.skills_dir or not sites:
            return ""

        descriptions = []
        descriptions.append("# Available Compound Skills\n")
        descriptions.append(
            "You have access to the following compound skills that combine multiple actions:\n"
        )

        for site in sites:
            if site in self.skill_metadata and self.skill_metadata[site]:
                descriptions.append(
                    f"\n## {site.replace('_', ' ').title()} Skills\n"
                )

                for skill_name, metadata in self.skill_metadata[site].items():
                    # Format parameters
                    params = []
                    for p in metadata["parameters"]:
                        param_str = f"{p['name']}: {p['type']}"
                        if p["default"] is not None:
                            param_str += f" = {p['default']}"
                        params.append(param_str)

                    param_signature = ", ".join(params)

                    descriptions.append(f"### skill[{skill_name}]")
                    if params:
                        descriptions.append(
                            f"**Parameters:** {param_signature}"
                        )
                    descriptions.append(
                        f"**Description:** {metadata['description']}"
                    )
                    descriptions.append("")

        if len(descriptions) <= 2:  # Only header lines
            return ""

        descriptions.append("\n## How to Use Skills\n")
        descriptions.append(
            "To use a skill, output: ```skill[skill_name, param1=value1, param2=value2]```"
        )
        descriptions.append(
            'Example: ```skill[search_admin_reviews_by_keyword, keyword="satisfied"]```'
        )
        descriptions.append(
            "\nSkills are compound actions that combine multiple basic actions. Use them when appropriate to simplify your task."
        )

        return "\n".join(descriptions)

    def parse_skill_call(
        self, text: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse a skill call from agent output.

        Expected format: skill[skill_name, param1=value1, param2=value2]

        Args:
            text: Agent output text

        Returns:
            Tuple of (skill_name, parameters) or None if no skill call found
        """
        # Look for skill calls in the format: skill[name, param=value, ...]
        pattern = r"skill\[([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:,\s*(.+?))?\]"

        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return None

        # Use the last match (most recent skill call)
        match = matches[-1]
        skill_name = match.group(1)
        params_str = match.group(2)

        # Parse parameters
        parameters = {}
        if params_str:
            try:
                # Split by commas, but respect quotes
                # Use a simple parser for key=value pairs
                parts = []
                current = []
                in_quotes = False
                quote_char = None

                for char in params_str + ",":
                    if char in ('"', "'") and (
                        not in_quotes or char == quote_char
                    ):
                        in_quotes = not in_quotes
                        quote_char = char if in_quotes else None
                        current.append(char)
                    elif char == "," and not in_quotes:
                        if current:
                            parts.append("".join(current).strip())
                            current = []
                    else:
                        current.append(char)

                # Parse each key=value pair
                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Try to evaluate the value
                        try:
                            # Handle strings with quotes
                            if value.startswith(('"', "'")) and value.endswith(
                                ('"', "'")
                            ):
                                parameters[key] = value[1:-1]
                            else:
                                # Try to parse as Python literal
                                parameters[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If parsing fails, use as string
                            parameters[key] = value

            except Exception as e:
                logger.warning(
                    f"Failed to parse skill parameters: {params_str}, error: {e}"
                )
                return None

        return skill_name, parameters

    async def execute_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        page: Any,
        sites: List[str],
    ) -> Any:
        """
        Execute a skill function.

        Args:
            skill_name: Name of the skill to execute
            parameters: Parameters to pass to the skill
            page: Playwright page object
            sites: List of sites for this task

        Returns:
            Result from the skill function
        """
        # Find the skill in available sites
        skill_func = None
        for site in sites:
            if (
                site in self.loaded_skills
                and skill_name in self.loaded_skills[site]
            ):
                skill_func = self.loaded_skills[site][skill_name]
                logger.debug(f"Found skill {skill_name} in site {site}")
                break

        if not skill_func:
            available = {
                site: list(self.loaded_skills.get(site, {}).keys())
                for site in sites
            }
            raise ValueError(
                f"Skill '{skill_name}' not found for sites: {sites}. Available skills: {available}"
            )

        # Execute the skill
        try:
            logger.info(
                f"Executing skill: {skill_name} with params: {parameters}"
            )
            logger.debug(
                f"skill_func type: {type(skill_func)}, is_coroutine: {inspect.iscoroutinefunction(skill_func)}"
            )

            # Call the skill function (handles both sync and async)
            if inspect.iscoroutinefunction(skill_func):
                # Async function - await it
                coro = skill_func(page, **parameters)
                logger.debug(f"Call result type: {type(coro)}")

                if coro is None:
                    raise TypeError(
                        f"Skill function {skill_name} returned None instead of a coroutine"
                    )

                result = await coro
            else:
                # Sync function - just call it
                result = skill_func(page, **parameters)

            logger.info(f"Skill {skill_name} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error executing skill {skill_name}: {e}")
            raise

    def has_skills_for_sites(self, sites: List[str]) -> bool:
        """Check if any skills are available for the given sites."""
        for site in sites:
            if site in self.skill_metadata and self.skill_metadata[site]:
                return True
        return False
