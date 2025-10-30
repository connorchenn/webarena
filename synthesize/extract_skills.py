#!/usr/bin/env python3
"""
Extract synthesized skills from JSON and organize by site for run.py integration.

Usage:
    # Single Python file:
    python extract_skills_to_py.py synthesized_skills.json --output skills.py

    # Organized by site for run.py:
    python extract_skills_to_py.py synthesized_skills.json --output-dir skills/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_to_directory(skills: dict, output_dir: str):
    """Organize skills by site into directory structure for run.py."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group skills by site
    skills_by_site = defaultdict(list)
    for skill_name, skill_data in skills.items():
        site = skill_data["site"]
        skills_by_site[site].append((skill_name, skill_data))

    # Create a directory for each site
    for site, site_skills in skills_by_site.items():
        site_dir = output_path / site
        site_dir.mkdir(exist_ok=True)

        # Create __init__.py to make it a package
        init_file = site_dir / "__init__.py"
        with open(init_file, "w") as f:
            f.write(f'"""Skills for {site} site"""\n')

        # Write each skill to its own file
        for skill_name, skill_data in site_skills:
            skill_file = site_dir / f"{skill_name}.py"

            lines = [
                '"""',
                f"Skill: {skill_name}",
                f"Site: {skill_data['site']}",
                f"Description: {skill_data['description']}",
                f"Synthesized from {skill_data['num_examples']} examples",
                f"Reasoning: {skill_data['reasoning']}",
                '"""',
                "",
                "from playwright.async_api import Page",
                "from typing import Optional, Union",
                "",
                "",
                skill_data["code"],
                "",
            ]

            with open(skill_file, "w") as f:
                f.write("\n".join(lines))

            print(f"  ✓ Created {site}/{skill_name}.py")

    print(f"\n✓ Extracted {len(skills)} skills to {output_dir}")
    print(f"✓ Organized into {len(skills_by_site)} site directories")
    print(f"\nUsage with run.py:")
    print(f"  python run.py --skills_dir {output_dir} ...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract synthesized skills from JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Directory structure for run.py:
  python extract_skills_to_py.py synthesized_skills.json --output-dir skills/
        """,
    )
    parser.add_argument(
        "json_file", help="Input JSON file with synthesized skills"
    )

    output_group = parser.add_mutually_exclusive_group(required=False)

    output_group.add_argument(
        "--output-dir",
        "-d",
        help="Output as directory structure organized by site (for run.py)",
        default="skills/",
    )

    args = parser.parse_args()

    if not Path(args.json_file).exists():
        print(f"Error: {args.json_file} not found")
        return 1

    # Load skills
    with open(args.json_file, "r") as f:
        skills = json.load(f)

    if not skills:
        print("Warning: No skills found in JSON file")
        return 0

    extract_to_directory(skills, args.output_dir)

    # Print summary
    print(f"\nSkills summary:")
    for skill_name, skill_data in skills.items():
        params = [
            f"{p['name']}: {p['type']}"
            for p in skill_data["parameters"]
            if p["name"] != "page"
        ]
        param_str = ", ".join(params) if params else ""
        print(f"  - {skill_name}({param_str})")
        print(f"    Site: {skill_data['site']}")

    return 0


if __name__ == "__main__":
    exit(main())
