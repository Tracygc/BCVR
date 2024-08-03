# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import Optional
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictStr, validator
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictStr, validator
from lightly.openapi_generated.swagger_client.models.selection_strategy_type_v3 import SelectionStrategyTypeV3

class SelectionConfigV4EntryStrategyAllOf(BaseModel):
    """
    SelectionConfigV4EntryStrategyAllOf
    """
    type: SelectionStrategyTypeV3 = Field(...)
    distribution: Optional[StrictStr] = Field(None, description="The distribution of the balance selection strategy. If TARGET is selected, the target prop of the selection strategy needs to be set as well.")
    __properties = ["type", "distribution"]

    @validator('distribution')
    def distribution_validate_enum(cls, value):
        """Validates the enum"""
        if value is None:
            return value

        if value not in ('TARGET', 'UNIFORM', 'INPUT'):
            raise ValueError("must be one of enum values ('TARGET', 'UNIFORM', 'INPUT')")
        return value

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> SelectionConfigV4EntryStrategyAllOf:
        """Create an instance of SelectionConfigV4EntryStrategyAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigV4EntryStrategyAllOf:
        """Create an instance of SelectionConfigV4EntryStrategyAllOf from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SelectionConfigV4EntryStrategyAllOf.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SelectionConfigV4EntryStrategyAllOf) in the input: " + str(obj))

        _obj = SelectionConfigV4EntryStrategyAllOf.parse_obj({
            "type": obj.get("type"),
            "distribution": obj.get("distribution")
        })
        return _obj

