import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    StateInfo,
    action2str,
)
from browser_env.actions import _id2key, _id2role

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def _parse_accessibility_tree_name(node_text: str) -> str:
    """Parse the name from accessibility tree node text.

    Format: [ID] ROLE 'NAME' properties...
    Example: [12] textbox 'Search' focused: True required: False
    Returns: Search (without quotes)
    """
    # Match the pattern [ID] ROLE 'NAME' where NAME can contain any chars except single quote
    # or [ID] ROLE "NAME" for double quotes
    match = re.search(r"\[\d+\]\s+\w+\s+['\"]([^'\"]*)['\"]", node_text)
    if match:
        return match.group(1)
    return ""


def action2playwright_code(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
) -> str:
    """Convert an action to Playwright synchronous API code.

    This mirrors the logic in execute_action() from actions.py to generate
    the equivalent Playwright sync API code that would be executed.
    """

    match action["action_type"]:
        case ActionTypes.CLICK:
            # Priority: pw_code > element_id > element_role+name
            if action["pw_code"]:
                # If pw_code exists, use it directly
                return action["pw_code"]
            elif action["element_id"]:
                # Element ID uses coordinate-based clicking in execution
                # Try to get semantic information from accessibility tree for better code
                text_meta_data = observation_metadata.get("text", {})
                if action["element_id"] in text_meta_data.get(
                    "obs_nodes_info", {}
                ):
                    node_info = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]
                    node_text = node_info.get("text", "").strip()

                    # Parse the accessibility tree format: [ID] ROLE 'NAME' properties...
                    if node_text:
                        # Extract the role and name
                        element_name = _parse_accessibility_tree_name(
                            node_text
                        )

                        # Extract role (word after [ID])
                        role_match = re.search(r"\[\d+\]\s+(\w+)", node_text)
                        element_role = (
                            role_match.group(1) if role_match else ""
                        )

                        if element_name and element_role:
                            # Generate appropriate locator based on role
                            element_name_escaped = element_name.replace(
                                '"', '\\"'
                            )

                            if element_role == "textbox":
                                return f'page.get_by_role("textbox", name="{element_name_escaped}").click()'
                            elif element_role == "button":
                                return f'page.get_by_role("button", name="{element_name_escaped}").click()'
                            elif element_role == "link":
                                return f'page.get_by_role("link", name="{element_name_escaped}").click()'
                            else:
                                return f'page.get_by_role("{element_role}", name="{element_name_escaped}").click()'

                # If we can't get semantic info, indicate coordinate-based clicking
                return (
                    f'# Click element [{action["element_id"]}] via coordinates'
                )

            elif action["element_role"] and action["element_name"]:
                # Use role-based locator with name
                element_role = _id2role[action["element_role"]]
                element_name = action["element_name"].replace('"', '\\"')

                # Handle special locator types
                if element_role == "alt_text":
                    return f'page.get_by_alt_text("{element_name}").click()'
                elif element_role == "label":
                    return f'page.get_by_label("{element_name}").click()'
                elif element_role == "placeholder":
                    return f'page.get_by_placeholder("{element_name}").click()'
                else:
                    return f'page.get_by_role("{element_role}", name="{element_name}").click()'

            return "# No locator information available for click"

        case ActionTypes.TYPE:
            text = "".join([_id2key[i] for i in action["text"]])
            text_escaped = text.replace('"', '\\"').replace("\n", "\\n")

            # Priority: pw_code > element_id > element_role+name
            if action["pw_code"]:
                return action["pw_code"]
            elif action["element_id"]:
                # Element ID: click element first, then type
                text_meta_data = observation_metadata.get("text", {})
                if action["element_id"] in text_meta_data.get(
                    "obs_nodes_info", {}
                ):
                    node_info = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]
                    node_text = node_info.get("text", "").strip()

                    # Parse the accessibility tree format: [ID] ROLE 'NAME' properties...
                    if node_text:
                        # Extract the role and name
                        element_name = _parse_accessibility_tree_name(
                            node_text
                        )

                        # Extract role (word after [ID])
                        role_match = re.search(r"\[\d+\]\s+(\w+)", node_text)
                        element_role = (
                            role_match.group(1) if role_match else ""
                        )

                        if element_name and element_role:
                            # Generate appropriate locator based on role
                            element_name_escaped = element_name.replace(
                                '"', '\\"'
                            )

                            if element_role == "textbox":
                                return f'page.get_by_role("textbox", name="{element_name_escaped}").fill("{text_escaped}")'
                            elif element_role == "searchbox":
                                return f'page.get_by_role("searchbox", name="{element_name_escaped}").fill("{text_escaped}")'
                            else:
                                return f'page.get_by_role("{element_role}", name="{element_name_escaped}").fill("{text_escaped}")'

                # If we can't get semantic info, show the actual execution pattern
                return f'# Click element [{action["element_id"]}], then: page.keyboard.type("{text_escaped}")'

            elif action["element_role"] and action["element_name"]:
                # Use role-based locator with name
                element_role = _id2role[action["element_role"]]
                element_name = action["element_name"].replace('"', '\\"')

                # Handle special locator types
                if element_role == "alt_text":
                    return f'page.get_by_alt_text("{element_name}").fill("{text_escaped}")'
                elif element_role == "label":
                    return f'page.get_by_label("{element_name}").fill("{text_escaped}")'
                elif element_role == "placeholder":
                    return f'page.get_by_placeholder("{element_name}").fill("{text_escaped}")'
                else:
                    return f'page.get_by_role("{element_role}", name="{element_name}").fill("{text_escaped}")'

            # No element specified, just type
            return f'page.keyboard.type("{text_escaped}")'

        case ActionTypes.HOVER:
            # Priority: pw_code > element_id > element_role+name
            if action["pw_code"]:
                return action["pw_code"]
            elif action["element_id"]:
                # Element ID uses coordinate-based hovering
                text_meta_data = observation_metadata.get("text", {})
                if action["element_id"] in text_meta_data.get(
                    "obs_nodes_info", {}
                ):
                    node_info = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]
                    node_text = node_info.get("text", "").strip()

                    # Parse the accessibility tree format: [ID] ROLE 'NAME' properties...
                    if node_text:
                        # Extract the role and name
                        element_name = _parse_accessibility_tree_name(
                            node_text
                        )

                        # Extract role (word after [ID])
                        role_match = re.search(r"\[\d+\]\s+(\w+)", node_text)
                        element_role = (
                            role_match.group(1) if role_match else ""
                        )

                        if element_name and element_role:
                            # Generate appropriate locator based on role
                            element_name_escaped = element_name.replace(
                                '"', '\\"'
                            )
                            return f'page.get_by_role("{element_role}", name="{element_name_escaped}").hover()'

                return (
                    f'# Hover element [{action["element_id"]}] via coordinates'
                )

            elif action["element_role"] and action["element_name"]:
                element_role = _id2role[action["element_role"]]
                element_name = action["element_name"].replace('"', '\\"')

                if element_role == "alt_text":
                    return f'page.get_by_alt_text("{element_name}").hover()'
                elif element_role == "label":
                    return f'page.get_by_label("{element_name}").hover()'
                elif element_role == "placeholder":
                    return f'page.get_by_placeholder("{element_name}").hover()'
                else:
                    return f'page.get_by_role("{element_role}", name="{element_name}").hover()'

            return "# No locator information available for hover"

        case ActionTypes.SCROLL:
            # Scroll uses page.evaluate() to scroll by viewport height
            direction = action["direction"]
            if "up" in direction.lower():
                return 'page.evaluate("(document.scrollingElement || document.body).scrollTop -= window.innerHeight")'
            else:
                return 'page.evaluate("(document.scrollingElement || document.body).scrollTop += window.innerHeight")'

        case ActionTypes.KEY_PRESS:
            key_comb = action["key_comb"]
            return f'page.keyboard.press("{key_comb}")'

        case ActionTypes.GOTO_URL:
            url = action["url"]
            return f'page.goto("{url}")'

        case ActionTypes.NEW_TAB:
            return "context.new_page()"

        case ActionTypes.PAGE_CLOSE:
            return "page.close()"

        case ActionTypes.GO_BACK:
            return "page.go_back()"

        case ActionTypes.GO_FORWARD:
            return "page.go_forward()"

        case ActionTypes.PAGE_FOCUS:
            page_num = action["page_number"]
            return f"context.pages[{page_num}].bring_to_front()"

        case ActionTypes.CHECK:
            if action["pw_code"]:
                return action["pw_code"]
            return "# Check action requires pw_code"

        case ActionTypes.SELECT_OPTION:
            if action["pw_code"]:
                return action["pw_code"]
            return "# Select option requires pw_code"

        case ActionTypes.STOP:
            answer = action["answer"]
            return f"# STOP: {answer}"

        case ActionTypes.NONE:
            return "# No action"

        case ActionTypes.MOUSE_CLICK:
            # Low-level coordinate click
            left, top = action["coords"]
            return f"page.mouse.click({left} * viewport_width, {top} * viewport_height)"

        case ActionTypes.MOUSE_HOVER:
            # Low-level coordinate hover
            left, top = action["coords"]
            return f"page.mouse.move({left} * viewport_width, {top} * viewport_height)"

        case ActionTypes.KEYBOARD_TYPE:
            # Low-level keyboard typing
            text = "".join([_id2key[i] for i in action["text"]])
            text_escaped = text.replace('"', '\\"').replace("\n", "\\n")
            return f'page.keyboard.type("{text_escaped}")'

        case _:
            return f'# Unknown action type: {action["action_type"]}'


def get_render_action(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]["text"]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

            # Add Playwright synchronous API code
            pw_code = action2playwright_code(action, observation_metadata)
            action_str += f"<div class='playwright_code' style='background-color:lightblue'><pre><strong>Playwright Sync API:</strong>\n{pw_code}</pre></div>"

        case "playwright":
            action_str = action["pw_code"]
        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")
    return action_str


def get_action_description(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    prompt_constructor: PromptConstructor | None,
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    node_content = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]["text"]
                    node_content = " ".join(node_content.split()[1:])
                    action_str = action2str(
                        action, action_set_tag, node_content
                    )
                else:
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "playwright":
            action_str = action["pw_code"]

        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, config_file: str, result_dir: str, action_set_tag: str
    ) -> None:
        with open(config_file, "r") as f:
            _config = json.load(f)
            _config_str = ""
            for k, v in _config.items():
                _config_str += f"{k}: {v}\n"
            _config_str = f"<pre>{_config_str}</pre>\n"
            task_id = _config["task_id"]

        self.action_set_tag = action_set_tag

        self.render_file = open(
            Path(result_dir) / f"render_{task_id}.html", "a+"
        )
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"
        new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_obs = observation["image"]
            image = Image.fromarray(img_obs)  # type:ignore
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:50vw; height:auto;'/>\n"

        # meta data
        new_content += f"<div class='prev_action' style='background-color:pink'>{meta_data['action_history'][-1]}</div>\n"

        # action
        action_str = get_render_action(
            action,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()
