"""
Tests for the TrismikResponseMapper class.

This module tests the response mapping functionality that converts
JSON API responses to internal type objects.
"""

from trismik._mapper import TrismikResponseMapper
from trismik.types import TrismikProject


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
        assert (
            project.description is None
        )  # .get() should return None for missing key
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
        assert (
            project.description
            == 'Description with symbols: @#$%^&*()[]{}|\\:;"'
        )
        assert project.accountId == "org_special_123"
        assert project.createdAt == "2025-09-12T17:00:00.000Z"
        assert project.updatedAt == "2025-09-12T17:00:00.000Z"

    def test_should_handle_long_strings(self) -> None:
        """Test that the mapper handles long string values."""
        long_description = (
            "This is a very long description that contains many words and "
            "should test the mapper's ability to handle longer text content. "
            * 10
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
