"""
Utility for parsing structured prompt files.
Supports multiple prompts in a single file using section markers.
"""

from typing import Dict
import re


def load_prompts(filepath: str) -> Dict[str, str]:
    """
    Load prompts from a structured prompt file.
    
    Format:
        [SECTION_NAME]
        Prompt text here...
        Can be multiple lines.
        
        [ANOTHER_SECTION]
        Another prompt text...
    
    Args:
        filepath: Path to the prompt file
        
    Returns:
        Dictionary mapping section names (lowercase) to prompt text
        
    Example:
        prompts = load_prompts("prompts/<use_case_id>.txt")
        policy_prompt = prompts["policy"]
        analysis_prompt = prompts["analysis"]
    """
    prompts = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove comments (lines starting with #)
    lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
    content = '\n'.join(lines)
    
    # Split by section headers [SECTION_NAME]
    sections = re.split(r'\n\s*\[([A-Z_]+)\]\s*\n', content)
    
    # First element is content before any section (usually empty or metadata)
    # Then alternating section names and content
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            section_name = sections[i].lower()
            section_content = sections[i + 1].strip()
            if section_content:  # Only add non-empty sections
                prompts[section_name] = section_content
    
    return prompts


def load_prompt_section(filepath: str, section: str) -> str:
    """
    Load a specific section from a prompt file.
    
    Args:
        filepath: Path to the prompt file
        section: Section name (case-insensitive)
        
    Returns:
        The prompt text for that section
        
    Raises:
        KeyError: If section not found
        
    Example:
        policy = load_prompt_section("prompts/<use_case_id>.txt", "policy")
    """
    prompts = load_prompts(filepath)
    section_lower = section.lower()
    
    if section_lower not in prompts:
        available = ", ".join(prompts.keys())
        raise KeyError(f"Section '{section}' not found. Available sections: {available}")
    
    return prompts[section_lower]
