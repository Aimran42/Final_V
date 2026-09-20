from typing import Dict, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class FunctionParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "integer", "float", "boolean", "number"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode="after")
    def check_not_empty(self) -> "FunctionDefinition":
        if not self.name.strip() or not self.description.strip():
            raise ValueError("name and description must not be empty")
        if not self.returns.type.strip():
            raise ValueError("returns.type must not be empty")
        for key, param in self.parameters.items():
            if not key.strip() or not param.type.strip():
                raise ValueError("parameter keys and types must not be empty")
        return self


class TestPrompt(BaseModel):
    prompt: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_fields(self) -> "TestPrompt":
        if not self.prompt.strip():
            raise ValueError(
                "prompt field cannot be empty or contain only spaces"
            )
        return self


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
