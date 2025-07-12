import logging
from datetime import datetime
from typing import Dict

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_current_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().isoformat()

def format_probabilities(probabilities: list, classes: list) -> Dict[str, float]:
    """Format probabilities for response"""
    return {f"Stage_{cls}": float(prob) for cls, prob in zip(classes, probabilities)}