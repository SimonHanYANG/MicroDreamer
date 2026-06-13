"""Annotation format for language-conditioned training."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Subgoal:
    """A single subgoal in a task."""
    description: str
    start_frame: int
    end_frame: int
    action_type: str = ""  # e.g., "move_stage", "inject", "aspire"


@dataclass
class EpisodeAnnotation:
    """Full annotation for one episode."""
    episode_id: str
    task_description: str
    subgoals: List[Subgoal] = field(default_factory=list)
    num_frames: int = 0
    success: bool = True
    notes: str = ""

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EpisodeAnnotation":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        subgoals = [Subgoal(**sg) for sg in data.pop("subgoals", [])]
        return cls(subgoals=subgoals, **data)


# Predefined task templates
TASK_TEMPLATES = {
    "icsi": {
        "description": "Perform ICSI (intracytoplasmic sperm injection) on an oocyte",
        "subgoals": [
            {"description": "Locate and approach the sperm cell", "action_type": "move_stage"},
            {"description": "Aspirate the sperm cell", "action_type": "aspire"},
            {"description": "Move to the oocyte", "action_type": "move_stage"},
            {"description": "Inject sperm into oocyte", "action_type": "inject"},
        ],
    },
    "cell_sorting": {
        "description": "Sort target cells from a population",
        "subgoals": [
            {"description": "Identify target cell", "action_type": "observe"},
            {"description": "Approach target cell", "action_type": "move_stage"},
            {"description": "Aspirate target cell", "action_type": "aspire"},
            {"description": "Move to collection area", "action_type": "move_stage"},
            {"description": "Release cell", "action_type": "release"},
        ],
    },
    "embryo_transfer": {
        "description": "Transfer embryo to target location",
        "subgoals": [
            {"description": "Locate embryo", "action_type": "observe"},
            {"description": "Aspirate embryo gently", "action_type": "aspire"},
            {"description": "Navigate to target", "action_type": "move_stage"},
            {"description": "Release embryo", "action_type": "release"},
        ],
    },
}


def create_annotation(
    episode_id: str,
    task_type: str,
    num_frames: int,
    custom_description: str = None,
) -> EpisodeAnnotation:
    """Create annotation from a task template."""
    template = TASK_TEMPLATES.get(task_type)
    if template is None:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(TASK_TEMPLATES.keys())}")

    subgoals = [Subgoal(**sg, start_frame=0, end_frame=0) for sg in template["subgoals"]]
    return EpisodeAnnotation(
        episode_id=episode_id,
        task_description=custom_description or template["description"],
        subgoals=subgoals,
        num_frames=num_frames,
    )
