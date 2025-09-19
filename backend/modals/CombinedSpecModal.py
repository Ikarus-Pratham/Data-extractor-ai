from typing import Dict, Any
from pydantic import BaseModel, model_validator, Field

class CombinedSpec(BaseModel):
    name: str
    general_specification: Dict[str, Any]
    description: Dict[str, Any]
    colors: Dict[str, Any]
    pricing: Dict[str, Any]
    dimensions: Dict[str, Any]
    components: Dict[str, Any]
    adjustments: Dict[str, Any]
    extra_remaining_info: Dict[str, Any] = Field(default_factory=dict, alias="extra/remaining_info")

    class Config:
        populate_by_name = True

    @staticmethod
    def _ensure_obj(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"items": value}
        if isinstance(value, str):
            return {"text": value}
        return {"value": value}

    @model_validator(mode="before")
    def coerce_sections_to_objects(cls, data: Any):  # type: ignore
        if not isinstance(data, dict):
            return data
        section_keys = [
            "general_specification",
            "description",
            "colors",
            "pricing",
            "dimensions",
            "components",
            "adjustments",
            "extra_remaining_info",
            "extra/remaining_info",
        ]
        for key in section_keys:
            if key in data:
                data[key] = CombinedSpec._ensure_obj(data[key])
        # Ensure alias maps correctly if provider emits alias key
        if "extra/remaining_info" in data and "extra_remaining_info" not in data:
            data["extra_remaining_info"] = data.pop("extra/remaining_info")
        return data