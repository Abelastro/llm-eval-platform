"""Rubric definitions for LLM-as-judge scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RubricCriterion(BaseModel):
    """A single weighted criterion in a judge rubric.

    Attributes:
        name: Criterion identifier.
        description: Instruction describing what the criterion rewards.
        weight: Relative weight applied to the criterion score.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    weight: float = 1.0


class Rubric(BaseModel):
    """An ordered collection of weighted criteria.

    Attributes:
        name: Rubric identifier.
        criteria: The criteria applied by the judge.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    criteria: list[RubricCriterion] = Field(default_factory=list)

    def normalized_weights(self) -> dict[str, float]:
        """Return criterion weights normalized to sum to one.

        Returns:
            Mapping of criterion name to normalized weight; empty when the
            rubric has no criteria or zero total weight.
        """

        total = sum(criterion.weight for criterion in self.criteria)
        if total <= 0:
            return {}
        return {criterion.name: criterion.weight / total for criterion in self.criteria}

    def prompt(self) -> str:
        """Render the rubric as an instruction block for a judge model.

        Returns:
            A newline-delimited description of every criterion.
        """

        lines = [f"Rubric: {self.name}"]
        for criterion in self.criteria:
            lines.append(
                f"- {criterion.name} (weight {criterion.weight:g}): "
                f"{criterion.description}"
            )
        return "\n".join(lines)


def default_rubric() -> Rubric:
    """Return a general-purpose answer-quality rubric.

    Returns:
        A :class:`Rubric` with correctness, completeness, and clarity criteria.
    """

    return Rubric(
        name="general-answer-quality",
        criteria=[
            RubricCriterion(
                name="correctness",
                description="The answer is factually consistent with the expected answer.",
                weight=3.0,
            ),
            RubricCriterion(
                name="completeness",
                description="The answer covers the key points without omissions.",
                weight=2.0,
            ),
            RubricCriterion(
                name="clarity",
                description="The answer is well-structured and easy to read.",
                weight=1.0,
            ),
        ],
    )
