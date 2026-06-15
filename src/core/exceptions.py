from __future__ import annotations


class ForecastLabError(Exception):
    """Base application error with a user-facing message."""

    error_code = "FORECAST_LAB_ERROR"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ValidationError(ForecastLabError):
    error_code = "VALIDATION_ERROR"


class ResultNotFoundError(ForecastLabError):
    error_code = "RESULT_NOT_FOUND"
