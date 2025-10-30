#!/usr/bin/env python3
"""
Example of how to use synthesized skills in a WebArena agent.

This shows how the generated compound APIs can be integrated
into an agent to make it more efficient and capable.
"""

import os

from playwright.sync_api import sync_playwright

# Import synthesized skills (after running synthesize_skills.py)
# from synthesized_skills import (
#     search_reviews_by_keyword,
#     navigate_to_admin_section,
#     filter_and_count_items
# )


def example_agent_with_skills():
    """
    Example agent that uses synthesized compound skills.

    Without synthesized skills, the agent would need to:
    1. Generate each low-level action
    2. Execute them one by one
    3. Handle state management between actions

    With synthesized skills, the agent can:
    1. Call one high-level function
    2. Get the result directly
    3. Focus on high-level planning
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Example 1: Using a compound search skill
        # Instead of: click customers → click marketing → click reviews → type keyword → extract count
        # Just do: search_reviews_by_keyword(page, "satisfied", admin_url)

        admin_url = os.environ.get(
            "SHOPPING_ADMIN", "http://localhost:7780/admin"
        )

        print("Example 1: Searching reviews")
        print("Without skills: 5 separate agent decisions and actions")
        print("With skills: 1 function call")
        # count = search_reviews_by_keyword(page, "satisfied", admin_url)
        # print(f"Found {count} reviews mentioning 'satisfied'")

        # Example 2: Navigating admin sections
        # Instead of: multiple clicks and waits
        # Just do: navigate_to_admin_section(page, admin_url, ["CUSTOMERS", "MARKETING"])

        print("\nExample 2: Navigation")
        print("Without skills: Agent must navigate step by step")
        print("With skills: Direct path to destination")
        # navigate_to_admin_section(page, admin_url, ["CUSTOMERS", "MARKETING"])

        # Example 3: Combining skills for complex tasks
        print("\nExample 3: Complex task composition")
        print("Task: Compare review counts for 'satisfied' vs 'excellent'")
        # satisfied_count = search_reviews_by_keyword(page, "satisfied", admin_url)
        # excellent_count = search_reviews_by_keyword(page, "excellent", admin_url)
        # print(f"Satisfied: {satisfied_count}, Excellent: {excellent_count}")

        browser.close()


def example_skill_library_growth():
    """
    How the skill library grows over time.

    Run 1: Agent solves tasks 0-20
    → Synthesizes 5 compound skills

    Run 2: Agent uses those 5 skills to solve tasks 21-40
    → Can complete tasks faster
    → Synthesizes 3 more skills (building on the original 5)

    Run 3: Agent now has 8 skills
    → Even more efficient
    → Can tackle harder tasks

    The library continuously grows and improves!
    """
    print("Skill library evolution:")
    print("Run 1: 0 skills → solve tasks → generate 5 skills")
    print("Run 2: 5 skills → solve faster → generate 3 new skills")
    print("Run 3: 8 skills → solve even faster → generate 2 new skills")
    print("...")
    print("Run N: Large skill library → agent becomes very efficient")


def example_skill_composition():
    """
    Future feature: Skills that use other skills.

    This is not yet implemented, but shows the vision.
    """
    print("\nFuture: Skill composition")
    print("Basic skills: navigate, search, filter, extract")
    print(
        "Composite skill: analyze_sentiment (uses search + extract + aggregate)"
    )
    print(
        "Higher-level skill: generate_report (uses multiple composite skills)"
    )
    print("\nThis creates a hierarchy of increasingly powerful capabilities!")


if __name__ == "__main__":
    print("=" * 80)
    print("Synthesized Skills Usage Examples")
    print("=" * 80)
    print()

    print("NOTE: These examples show the concept.")
    print("To actually run them, first generate skills with:")
    print("  export OPENAI_API_KEY='your-key'")
    print(
        "  conda run -n webarena python synthesize_skills.py results/gpt-4-turbo-2024-04-09"
    )
    print()
    print("=" * 80)
    print()

    example_agent_with_skills()
    print("\n" + "=" * 80 + "\n")
    example_skill_library_growth()
    print("\n" + "=" * 80 + "\n")
    example_skill_composition()
    print("\n" + "=" * 80)
