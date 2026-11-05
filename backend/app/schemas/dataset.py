"""
Pydantic schemas for dataset inspection API endpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatasetStatusResponse(BaseModel):
    dataset_path: str
    exists: bool
    structure_type: str
    num_total_images: int
    num_readable_images: int
    num_corrupted_images: int
    has_inspection_report: bool
    last_inspected_at: Optional[str] = None
    class_counts: Dict[str, int] = Field(default_factory=dict)


class DatasetInspectResponse(BaseModel):
    status: str
    message: str
    summary: Dict[str, Any]


class DatasetReportResponse(BaseModel):
    summary: Dict[str, Any]
    markdown_report: str
    inventory_count: int
    corrupted_count: int
    duplicate_groups_count: int


class DuplicateGroupSchema(BaseModel):
    group_id: int
    type: str
    hash_value: str
    files: List[str]
