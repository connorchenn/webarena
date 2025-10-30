#!/usr/bin/env python3
"""
Simple skill synthesis script for WebArena traces.
Analyzes successful trajectories and generates compound API functions.

NOTE: This script requires openai>=1.0 for vision support.
      Run in the 'webarena-skills' conda environment.
"""

import ast
import base64
import json
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

import yaml
from extract_skills import extract_to_directory
from openai import OpenAI

# Try to import playwright, but don't fail if not available
from playwright.sync_api import Page, sync_playwright

# Maximum number of retries for code generation
MAX_RETRY = 3


def create_merged_log(results_dir):
    """Create merged_log.txt from log files."""
    log_file_path = Path(results_dir) / "log_files.txt"
    merged_log_path = Path(results_dir) / "merged_log.txt"

    if merged_log_path.exists():
        return True

    if not log_file_path.exists():
        print(f"Warning: No log_files.txt found in {results_dir}")
        return False

    with open(log_file_path, "r") as f:
        log_files = [line.strip() for line in f if line.strip()]

    # Merge all log files
    merged_content = []
    for log_file in log_files:
        # Try both absolute and relative paths
        log_path = Path(results_dir).parent.parent / log_file
        if not log_path.exists():
            log_path = Path(log_file)
        if not log_path.exists():
            continue

        with open(log_path, "r") as f:
            merged_content.extend(f.readlines())

    if merged_content:
        with open(merged_log_path, "w") as f:
            f.writelines(merged_content)
        print(f"✓ Created merged_log.txt")
        return True

    return False


def create_combined_config(results_dir):
    """Create a combined config file from individual task configs in config_files/."""
    combined_config_path = Path(results_dir) / "combined_config.json"

    if combined_config_path.exists():
        return combined_config_path

    # Find all render HTML files to know which task IDs we have
    results_path = Path(results_dir)
    render_files = sorted(results_path.glob("render_*.html"))
    task_ids = [int(f.stem.replace("render_", "")) for f in render_files]

    # Load individual config files
    configs = []
    config_dir = Path("config_files")

    for task_id in task_ids:
        config_file = config_dir / f"{task_id}.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
                configs.append(config)

    if configs:
        with open(combined_config_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"✓ Created combined config with {len(configs)} tasks")
        return combined_config_path

    return None


def convert_html_to_json(results_dir):
    """Convert HTML render files to JSON using html2json.py."""
    json_path = Path(results_dir) / "json_dump.json"

    # Check if JSON already exists
    if json_path.exists():
        print(f"JSON file already exists at {json_path}")
        return json_path

    # Create merged log file first
    if not create_merged_log(results_dir):
        print(
            "Warning: Could not create merged_log.txt, results may be incomplete"
        )

    # Create combined config file
    config_path = create_combined_config(results_dir)

    if not config_path:
        print("Warning: Could not create combined config file")
        config_arg = ""
    else:
        config_arg = f"--config_json {config_path}"

    print(f"Converting HTML to JSON...")
    cmd = f"python scripts/html2json.py --result_folder {results_dir} {config_arg}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running html2json.py:")
        print(result.stderr)
        return None

    if result.stdout:
        print(result.stdout)

    print(f"✓ Created {json_path}")
    return json_path


def get_function_info(code_string):
    """Extract function names and parameters from code using AST parsing."""
    tree = ast.parse(code_string)
    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            params = []

            # Get regular parameters
            for arg in node.args.args:
                params.append(arg.arg)

            functions[func_name] = params

    return functions


def validate_example_parameters(
    code: str, example_call: dict, function_name: str
) -> tuple[bool, str]:
    """Validate that all function parameters (except 'page') are present in example_call."""
    try:
        functions = get_function_info(code)
        if not functions:
            return False, "No functions found in code"

        # Get the specific function by name
        if function_name not in functions:
            return (
                False,
                f"Function '{function_name}' not found in code. Available functions: {list(functions.keys())}",
            )

        params = functions[function_name]

        # Filter out 'page' parameter
        non_page_params = [p for p in params if p != "page"]

        for param in example_call:
            if param not in non_page_params:
                return (
                    False,
                    f"Parameter {param} not found in function signature",
                )
        return True, "All parameters present in example_call"

    except Exception as e:
        return False, f"Error validating example parameters: {str(e)}"


def validate_code(code: str) -> bool:
    """Validate that the code compiles correctly using ast.parse."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  Code validation failed: {e}")
        return False
    except Exception as e:
        print(f"  Code validation error: {e}")
        return False


def validate_code_runtime(
    code: str,
    site: str,
    example_params: dict = None,
    function_name: str = None,
) -> tuple[bool, str]:
    """Validate that the code runs correctly in a real browser environment."""

    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            # Add necessary imports and wrap the function for testing
            # Prepare example parameters for the test
            example_params_str = str(example_params or {})
            function_name = function_name or "unknown_function"

            test_code = f"""
import sys
import os
import traceback
from pathlib import Path

# Add the webarena directory to the path so we can import browser_env
webarena_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(webarena_dir))

from playwright.sync_api import sync_playwright, Page
from browser_env.envs import ScriptBrowserEnv

# Site to config file mapping
SITE_TO_CONFIG_MAPPING = {{
    "shopping_admin": "0.json",
    "shopping": "0.json",
    "map": "10.json",
    "reddit": "27.json",
    "gitlab": "102.json",
    "wikipedia": "0.json",
    "homepage": "0.json"
}}

{code}

# Test function
def test_function():
    # Use the provided function name
    func_name = "{function_name}"

    # Get the function from globals
    exec_globals = {{}}
    exec('''{code}''', exec_globals)
    func = exec_globals[func_name]

    # Create a real browser environment
    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=0,
        observation_type="html",
        current_viewport_only=False,
        viewport_size={{"width": 1280, "height": 720}},
        save_trace_enabled=False,
        sleep_after_execution=0.0
    )

    try:
        # Find appropriate config file for the site
        config_file = SITE_TO_CONFIG_MAPPING["{site}"]
        config_path = f"config_files/{{config_file}}"

        # Reset the environment with site-specific config
        observation, info = env.reset(options={{"config_file": config_path}})
        page = env.page

        # Try to call the function with the real page and example parameters
        # This will test if the function can be called without errors
        # and if it can handle real web page interactions
        try:
            # Use example parameters if provided, otherwise call with just page
            example_params = {example_params_str}
            if example_params:
                func(page, **example_params)
            else:
                func(page)
            return True, f"Runtime validation passed with real browser environment (site: {site}, config: {{config_file}})"
        except Exception as e:
            # Capture the full traceback for better error reporting
            error_trace = traceback.format_exc()
            return False, f"Runtime error when executing function: {{str(e)}}\\nTraceback:\\n{{error_trace}}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return False, f"Runtime error during browser setup: {{str(e)}}\\nTraceback:\\n{{error_trace}}"
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    success, message = test_function()
    if not success:
        print(f"ERROR: {{message}}")
        exit(1)
    else:
        print(f"SUCCESS: {{message}}")
"""
            f.write(test_code)
            temp_file = f.name

        # Execute the test code
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout for real browser testing
        )

        # Clean up
        os.unlink(temp_file)

        if result.returncode == 0:
            return (
                True,
                f"Runtime validation passed with real browser environment (site: {site})",
            )
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            return False, f"Runtime error: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Runtime validation timed out (60s)"
    except Exception as e:
        return False, f"Runtime validation error: {str(e)}"
    finally:
        # Clean up temp file if it exists
        try:
            if "temp_file" in locals():
                os.unlink(temp_file)
        except:
            pass


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def prepare_trajectory_content(
    trajectories: List[Dict], config: Dict[str, Any], max_examples: int = None
) -> List[Dict]:
    """Prepare trajectory content with images for GPT vision API."""
    content = []

    # Add trajectory examples (limit to max_examples)
    for idx, traj in enumerate(trajectories[:max_examples]):
        # Clearly label whether trajectory was successful or not
        success_label = "✓ SUCCESSFUL" if traj["success"] else "✗ UNSUCCESSFUL"
        traj_text = f"\n\n=== TRAJECTORY {idx + 1} [{success_label}] ===\nTask: {traj['intent']}\n\n"
        content.append({"type": "text", "text": traj_text})

        # Add each interaction (screenshot + observation + action)
        for i, interaction in enumerate(traj["interactions"]):
            if "user" in interaction:
                content.append(
                    {
                        "type": "text",
                        "text": f"Step {i//2 + 1} Observation:\n{interaction['user']}\n",
                    }
                )

                # Add image if available
                if "image" in interaction:
                    image_path = interaction["image"]
                    if os.path.exists(image_path):
                        try:
                            image_data = encode_image_to_base64(image_path)
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}",
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not encode image {image_path}: {e}"
                            )

            elif "assistant" in interaction:
                content.append(
                    {
                        "type": "text",
                        "text": f"Action: {interaction['assistant']}\n",
                    }
                )

    return content


def synthesize_skill_from_trajectories(
    site: str,
    trajectories: List[Dict],
    client: OpenAI,
    config: Dict[str, Any],
    existing_skills: List[str] = [],
) -> Dict:
    """Use GPT-4o with vision to synthesize a compound API from multiple trajectories.

    Note: Requires openai>=1.0 for vision support.
    """

    if not trajectories:
        return None

    # Get max examples from config
    max_examples = config["synthesis"]["max_examples_per_site"]

    # Prepare content with images
    content = prepare_trajectory_content(
        trajectories, config, max_examples=max_examples
    )

    # Add existing skills context
    if existing_skills:
        existing_str = "\n".join([f"- {s}" for s in existing_skills])
        content.append(
            {
                "type": "text",
                "text": f"\n\nExisting skills you can build upon:\n{existing_str}",
            }
        )

    # Retry logic for code generation
    for attempt in range(MAX_RETRY):
        try:
            # Get model config
            model_config = config["model"]

            # Use new API with vision support
            response = client.chat.completions.create(
                model=model_config["name"],
                max_completion_tokens=model_config.get("max_tokens", 32000),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": content},
                ],
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            result = json.loads(response_text)

            if result.get("skip"):
                print(
                    f"  Skipped: {result.get('reason', 'Not useful as compound pattern')}"
                )
                return None

            # Validate required fields
            required = [
                "function_name",
                "description",
                "parameters",
                "code",
                "example_call",
            ]
            if not all(key in result for key in required):
                print(f"  Error: Missing required fields in response")
                print(f"  Got: {list(result.keys())}")
                if attempt < MAX_RETRY - 1:
                    print(f"  Retrying... (attempt {attempt + 1}/{MAX_RETRY})")
                    continue
                return None

            # Validate code compiles correctly
            if not validate_code(result["code"]):
                if attempt < MAX_RETRY - 1:
                    print(
                        f"  Code validation failed, retrying... (attempt {attempt + 1}/{MAX_RETRY})"
                    )
                    continue
                else:
                    print(
                        f"  Code validation failed after {MAX_RETRY} attempts, skipping"
                    )
                    return None

            print(f"  ✓ Code validation passed (attempt {attempt + 1})")

            # Validate example parameters match function signature
            param_valid, param_msg = validate_example_parameters(
                result["code"],
                result.get("example_call", {}),
                result["function_name"],
            )
            if not param_valid:
                print(f"  Parameter validation failed: {param_msg}")
                if attempt < MAX_RETRY - 1:
                    print(
                        f"  Retrying due to parameter mismatch... (attempt {attempt + 1}/{MAX_RETRY})"
                    )
                    # Add parameter validation error context to the next attempt
                    content.append(
                        {
                            "type": "text",
                            "text": f"\n\nPrevious attempt failed parameter validation:\n{param_msg}\n\nPlease ensure the example_call contains all required parameters with correct names.",
                        }
                    )
                    continue
                else:
                    print(
                        f"  Parameter validation failed after {MAX_RETRY} attempts, skipping"
                    )
                    return None

            print(f"  ✓ Parameter validation passed (attempt {attempt + 1})")

            # Validate code runs correctly in browser environment
            runtime_valid, runtime_msg = validate_code_runtime(
                result["code"],
                site,
                result["example_call"],
                result["function_name"],
            )
            if not runtime_valid:
                print(f"  Runtime validation failed: {runtime_msg}")
                if attempt < MAX_RETRY - 1:
                    print(
                        f"  Retrying due to runtime errors... (attempt {attempt + 1}/{MAX_RETRY})"
                    )
                    # Add runtime error context to the next attempt
                    content.append(
                        {
                            "type": "text",
                            "text": f"\n\nPrevious attempt failed with runtime error:\n{runtime_msg}\n\nPlease fix the code to avoid this error.",
                        }
                    )
                    continue
                else:
                    print(
                        f"  Runtime validation failed after {MAX_RETRY} attempts, skipping"
                    )
                    return None

            print(f"  ✓ Runtime validation passed (attempt {attempt + 1})")
            return result

        except json.JSONDecodeError as e:
            print(f"  Error parsing JSON response: {e}")
            print(f"  Response preview: {response_text[:300]}...")
            if attempt < MAX_RETRY - 1:
                print(f"  Retrying... (attempt {attempt + 1}/{MAX_RETRY})")
                continue
            return None
        except Exception as e:
            print(f"  Error during synthesis: {e}")
            if attempt < MAX_RETRY - 1:
                print(f"  Retrying... (attempt {attempt + 1}/{MAX_RETRY})")
                continue
            import traceback

            traceback.print_exc()
            return None

    # If we get here, all retries failed
    print(f"  Failed to generate valid code after {MAX_RETRY} attempts")
    return None


def load_trajectories_from_json(json_path):
    """Load all trajectories from the JSON dump file grouped by site."""
    with open(json_path, "r") as f:
        data = json.load(f)

    trajectories = {}
    for _, task_data in data.items():
        traj = {
            "intent": task_data["intent"],
            "task_num": task_data["task_id"],
            "success": task_data["success"],
            "interactions": task_data["messages"],
        }

        # Convert sites list to a string key
        # e.g., ['shopping_admin'] -> 'shopping_admin'
        # e.g., ['reddit', 'gitlab'] -> 'reddit,gitlab'
        # sites_key = ','.join(sorted(task_data["sites"]))
        sites_key = task_data["sites"][0]

        if sites_key not in trajectories:
            trajectories[sites_key] = []
        trajectories[sites_key].append(traj)

    return trajectories


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthesize skills from WebArena traces"
    )
    parser.add_argument(
        "results_dir",
        help="Results directory (e.g., results/gpt-4-turbo-2024-04-09)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--output_dir",
        default="skills_output",
        help="Output directory for skills",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize OpenAI client (new API with vision)
    api_key = os.environ.get("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # Convert HTML to JSON if needed
    json_path = os.path.join(args.results_dir, "json_dump.json")
    if not os.path.exists(json_path):
        print("\nConverting HTML traces to JSON...")
        convert_html_to_json(args.results_dir)

    # Load trajectories grouped by site
    print(f"\nLoading trajectories from {json_path}...")
    trajectories_by_site = load_trajectories_from_json(json_path)

    total_trajectories = 0

    print(f"Total trajectories: {total_trajectories}")

    # Synthesize skills for each site
    all_skills = {}

    for site, trajs in trajectories_by_site.items():
        # Separate successful and unsuccessful trajectories
        successful_trajs = [t for t in trajs if t["success"]]
        unsuccessful_trajs = [t for t in trajs if not t["success"]]

        min_examples = config["synthesis"]["min_examples_required"]
        max_examples = config["synthesis"]["max_examples_per_site"]

        if len(successful_trajs) < min_examples:
            print(
                f"\nSkipping {site}: only {len(successful_trajs)} successful trajectories"
            )
            continue

        # Prioritize successful trajectories first, then add unsuccessful ones up to max_examples
        prioritized_trajs = successful_trajs + unsuccessful_trajs
        selected_trajs = prioritized_trajs[:max_examples]

        num_successful = sum(1 for t in selected_trajs if t["success"])
        num_unsuccessful = len(selected_trajs) - num_successful

        print(f"Synthesizing compound API for: {site}")
        print(
            f"Using {len(selected_trajs)} trajectories ({num_successful} successful, {num_unsuccessful} unsuccessful)"
        )

        # Synthesize skill from multiple examples
        skill_result = synthesize_skill_from_trajectories(
            site=site,
            trajectories=selected_trajs,
            client=client,
            config=config,
            existing_skills=list(all_skills.keys()),
        )

        if skill_result:
            all_skills[skill_result["function_name"]] = {
                "site": site,
                "description": skill_result["description"],
                "parameters": skill_result["parameters"],
                "code": skill_result["code"],
                "reasoning": skill_result.get("reasoning", ""),
                "num_examples": len(selected_trajs),
                "num_successful": num_successful,
                "num_unsuccessful": num_unsuccessful,
            }
            print(f"✓ Created: {skill_result['function_name']}")
            print(f"  Description: {skill_result['description']}")
            print(f"  Parameters: {len(skill_result['parameters'])}")
        else:
            print(f"✗ No skill synthesized for {site}")

    # Write output
    if all_skills:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save skills.json to output directory
        skills_json_path = output_dir / "skills.json"
        with open(skills_json_path, "w") as f:
            json.dump(all_skills, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Successfully synthesized {len(all_skills)} compound APIs")
        print(f"✓ Saved to: {skills_json_path}")

        print("Generated compound APIs:")
        for name, skill in all_skills.items():
            print(f"\n  {name}:")
            print(f"    Site: {skill['site']}")
            print(f"    Description: {skill['description']}")
            print(
                f"    Examples used: {skill['num_examples']} ({skill['num_successful']} successful, {skill['num_unsuccessful']} unsuccessful)"
            )

        # Automatically extract skills to Python files
        print(f"\n{'='*80}")
        print("Extracting skills to Python files...")
        print(f"{'='*80}\n")

        skills_dir = output_dir / "skills"
        extract_to_directory(all_skills, str(skills_dir))

        print(f"\n{'='*80}")
        print(f"✓ Skills ready for use with run.py:")
        print(f"  python run.py --skills_dir {skills_dir} ...")
        print(f"{'='*80}")
    else:
        print("\n✗ No compound APIs were synthesized")
        print("   This could mean:")
        print("   - Not enough successful trajectories per site")
        print("   - Trajectories don't show useful compound patterns")
        print(
            f"   - Need at least {config['synthesis']['min_examples_required']} successful examples per site"
        )


if __name__ == "__main__":
    main()
