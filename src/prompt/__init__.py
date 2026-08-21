"""
Prompt-based scene conditioning for FlowChain.

PromptGenerator: condition + domain → prefix tokens → FlowChain encoder
"""
from .prompt_generator import PromptGenerator
