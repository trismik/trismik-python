"""
Tests for the TrismikResponseMapper class.

This module tests the response mapping functionality that converts
JSON API responses to internal type objects.
"""

from trismik._mapper import TrismikResponseMapper
from trismik.types import (
    TrismikMultipleChoiceTextItem,
    TrismikOpenEndedTextItem,
    TrismikProject,
    TrismikRunSummary,
)


class TestTrismikResponseMapper:
    """Test suite for the TrismikResponseMapper class."""

    def test_should_map_project_with_all_fields(self) -> None:
        """Test mapping a complete project JSON response."""
        json_data = {
            "id": "project123",
            "name": "Test Project",
            "description": "A comprehensive test project",
            "accountId": "org456",
            "createdAt": "2025-09-12T10:30:00.000Z",
            "updatedAt": "2025-09-12T11:45:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project123"
        assert project.name == "Test Project"
        assert project.description == "A comprehensive test project"
        assert project.accountId == "org456"
        assert project.createdAt == "2025-09-12T10:30:00.000Z"
        assert project.updatedAt == "2025-09-12T11:45:00.000Z"

    def test_should_map_project_with_null_description(self) -> None:
        """Test mapping project JSON response with null description."""
        json_data = {
            "id": "project789",
            "name": "Project No Description",
            "description": None,
            "accountId": "org999",
            "createdAt": "2025-09-12T12:00:00.000Z",
            "updatedAt": "2025-09-12T12:00:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project789"
        assert project.name == "Project No Description"
        assert project.description is None
        assert project.accountId == "org999"
        assert project.createdAt == "2025-09-12T12:00:00.000Z"
        assert project.updatedAt == "2025-09-12T12:00:00.000Z"

    def test_should_map_project_with_missing_description_key(self) -> None:
        """Test mapping project JSON response with missing description key."""
        json_data = {
            "id": "project456",
            "name": "Project Missing Desc Key",
            "accountId": "org777",
            "createdAt": "2025-09-12T13:00:00.000Z",
            "updatedAt": "2025-09-12T13:00:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project456"
        assert project.name == "Project Missing Desc Key"
        assert project.description is None  # .get() should return None for missing key
        assert project.accountId == "org777"
        assert project.createdAt == "2025-09-12T13:00:00.000Z"
        assert project.updatedAt == "2025-09-12T13:00:00.000Z"

    def test_should_map_project_with_empty_string_description(self) -> None:
        """Test mapping project JSON response with empty string description."""
        json_data = {
            "id": "project001",
            "name": "Project Empty Desc",
            "description": "",
            "accountId": "org888",
            "createdAt": "2025-09-12T14:00:00.000Z",
            "updatedAt": "2025-09-12T14:00:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project001"
        assert project.name == "Project Empty Desc"
        assert project.description == ""  # Empty string should be preserved
        assert project.accountId == "org888"
        assert project.createdAt == "2025-09-12T14:00:00.000Z"
        assert project.updatedAt == "2025-09-12T14:00:00.000Z"

    def test_should_handle_different_date_formats(self) -> None:
        """Test that the mapper handles different date string formats."""
        # Test with different but valid ISO format
        json_data = {
            "id": "project_iso",
            "name": "ISO Date Project",
            "description": "Testing ISO dates",
            "accountId": "org_iso",
            "createdAt": "2025-09-12T15:30:45.123456Z",
            "updatedAt": "2025-09-12T16:45:30.987654Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project_iso"
        assert project.name == "ISO Date Project"
        assert project.description == "Testing ISO dates"
        assert project.accountId == "org_iso"
        assert project.createdAt == "2025-09-12T15:30:45.123456Z"
        assert project.updatedAt == "2025-09-12T16:45:30.987654Z"

    def test_should_preserve_special_characters_in_strings(self) -> None:
        """Test that special characters are preserved in string fields."""
        json_data = {
            "id": "project_special",
            "name": "Test Project with Special Chars: éñ中文",
            "description": 'Description with symbols: @#$%^&*()[]{}|\\:;"',
            "accountId": "org_special_123",
            "createdAt": "2025-09-12T17:00:00.000Z",
            "updatedAt": "2025-09-12T17:00:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project_special"
        assert project.name == "Test Project with Special Chars: éñ中文"
        assert project.description == 'Description with symbols: @#$%^&*()[]{}|\\:;"'
        assert project.accountId == "org_special_123"
        assert project.createdAt == "2025-09-12T17:00:00.000Z"
        assert project.updatedAt == "2025-09-12T17:00:00.000Z"

    def test_should_handle_long_strings(self) -> None:
        """Test that the mapper handles long string values."""
        long_description = (
            "This is a very long description that contains many words and "
            "should test the mapper's ability to handle longer text content. " * 10
        )

        json_data = {
            "id": "project_long",
            "name": "Project with Long Description",
            "description": long_description,
            "accountId": "org_long",
            "createdAt": "2025-09-12T18:00:00.000Z",
            "updatedAt": "2025-09-12T18:00:00.000Z",
        }

        project = TrismikResponseMapper.to_project(json_data)

        assert isinstance(project, TrismikProject)
        assert project.id == "project_long"
        assert project.name == "Project with Long Description"
        assert project.description == long_description
        assert project.accountId == "org_long"
        assert project.createdAt == "2025-09-12T18:00:00.000Z"
        assert project.updatedAt == "2025-09-12T18:00:00.000Z"

    def test_should_map_multiple_choice_text_item(self) -> None:
        """Test mapping a multiple choice text item with choices."""
        json_data = {
            "id": "item_1",
            "question": "What is 2+2?",
            "choices": [
                {"id": "A", "value": "3"},
                {"id": "B", "value": "4"},
            ],
        }

        item = TrismikResponseMapper.to_item(json_data)

        assert isinstance(item, TrismikMultipleChoiceTextItem)
        assert item.id == "item_1"
        assert item.question == "What is 2+2?"
        assert len(item.choices) == 2
        assert item.choices[0].id == "A"
        assert item.choices[0].text == "3"
        assert item.choices[1].id == "B"
        assert item.choices[1].text == "4"

    def test_should_map_open_ended_text_item_with_null_choices(self) -> None:
        """Test mapping an open-ended text item with choices: null."""
        json_data = {
            "id": "item_2",
            "question": "Explain gravity.",
            "choices": None,
        }

        item = TrismikResponseMapper.to_item(json_data)

        assert isinstance(item, TrismikOpenEndedTextItem)
        assert item.id == "item_2"
        assert item.question == "Explain gravity."
        assert item.reference is None
        assert item.responseText is None

    def test_should_map_open_ended_text_item_without_choices_key(self) -> None:
        """Test mapping an open-ended text item without choices key."""
        json_data = {
            "id": "item_3",
            "question": "Describe photosynthesis.",
        }

        item = TrismikResponseMapper.to_item(json_data)

        assert isinstance(item, TrismikOpenEndedTextItem)
        assert item.id == "item_3"
        assert item.question == "Describe photosynthesis."
        assert item.reference is None
        assert item.responseText is None

    def test_should_map_open_ended_text_item_with_reference_and_response(self) -> None:
        """Test mapping an open-ended text item with reference and responseText."""
        json_data = {
            "id": "item_4",
            "question": "What is photosynthesis?",
            "choices": None,
            "reference": "expected answer",
            "responseText": "model response",
        }

        item = TrismikResponseMapper.to_item(json_data)

        assert isinstance(item, TrismikOpenEndedTextItem)
        assert item.id == "item_4"
        assert item.question == "What is photosynthesis?"
        assert item.reference == "expected answer"
        assert item.responseText == "model response"

    def test_should_map_run_summary_with_dataset_item_type(self) -> None:
        """Test mapping run summary with datasetItemType."""
        json_data = {
            "id": "run_id",
            "datasetId": "test_id",
            "datasetItemType": "open_ended_text",
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "dataset": [],
            "responses": [],
            "metadata": {},
        }

        summary = TrismikResponseMapper.to_run_summary(json_data)

        assert isinstance(summary, TrismikRunSummary)
        assert summary.dataset_item_type == "open_ended_text"

    def test_should_map_run_summary_without_dataset_item_type(self) -> None:
        """Test mapping run summary without datasetItemType for backward compat."""
        json_data = {
            "id": "run_id",
            "datasetId": "test_id",
            "state": {
                "responses": [],
                "thetas": [],
                "std_error_history": [],
                "kl_info_history": [],
                "effective_difficulties": [],
            },
            "dataset": [],
            "responses": [],
            "metadata": {},
        }

        summary = TrismikResponseMapper.to_run_summary(json_data)

        assert isinstance(summary, TrismikRunSummary)
        assert summary.dataset_item_type is None
