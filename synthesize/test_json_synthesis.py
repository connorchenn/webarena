#!/usr/bin/env python3
"""Test script to verify JSON conversion and extraction works."""

import sys
from pathlib import Path

# Import functions from synthesize_skills
from synthesize_skills import (
    create_merged_log,
    convert_html_to_json,
    load_trajectories_from_json
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_json_synthesis.py <results_dir>")
        print("Example: python test_json_synthesis.py results/gpt-4-turbo-2024-04-09")
        return
    
    results_dir = sys.argv[1]
    
    print("="*80)
    print("Testing JSON-based Skill Synthesis Pipeline")
    print("="*80)
    print()
    
    # Step 1: Create merged log
    print("Step 1: Creating merged_log.txt...")
    if create_merged_log(results_dir):
        print("✓ Success")
    else:
        print("✗ Failed")
    print()
    
    # Step 2: Convert HTML to JSON
    print("Step 2: Converting HTML to JSON...")
    json_path = convert_html_to_json(results_dir)
    if json_path:
        print(f"✓ Success: {json_path}")
    else:
        print("✗ Failed")
        return
    print()
    
    # Step 3: Load trajectories
    print("Step 3: Loading trajectories from JSON...")
    trajectories = load_trajectories_from_json(json_path)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    print()
    
    # Step 4: Analyze trajectories
    print("Step 4: Analyzing trajectories...")
    print("="*80)
    
    passed_count = 0
    synthesizable_count = 0
    
    for traj in sorted(trajectories, key=lambda x: x['task_num']):
        task_num = traj['task_num']
        passed = traj['success']
        num_actions = len(traj['actions'])
        
        if passed:
            passed_count += 1
        
        if passed and num_actions >= 3:
            synthesizable_count += 1
            status = "✓ SYNTHESIZABLE"
        elif passed:
            status = "✓ PASS (< 3 actions)"
        else:
            status = "✗ FAIL"
        
        print(f"\nTask {task_num}: {status}")
        print(f"  Intent: {traj['intent'][:70]}...")
        print(f"  Actions: {num_actions}")
        
        if passed and num_actions >= 3:
            print("  First 5 actions:")
            for i, action in enumerate(traj['actions'][:5], 1):
                action_preview = action[:60] + "..." if len(action) > 60 else action
                print(f"    {i}. {action_preview}")
            if num_actions > 5:
                print(f"    ... and {num_actions - 5} more actions")
    
    print()
    print("="*80)
    print("SUMMARY:")
    print(f"  Total tasks: {len(trajectories)}")
    print(f"  Passed tasks: {passed_count}")
    print(f"  Synthesizable (PASS + 3+ actions): {synthesizable_count}")
    print("="*80)
    print()
    
    if synthesizable_count > 0:
        print("These trajectories are ready for GPT-5 synthesis!")
        print("To synthesize skills, run:")
        print(f"  export OPENAI_API_KEY='your-key'")
        print(f"  conda run -n webarena python synthesize_skills.py {results_dir}")
    else:
        print("No trajectories suitable for synthesis found.")
        print("(Need passed tasks with 3+ actions)")


if __name__ == '__main__':
    main()

