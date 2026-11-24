"""Categories for SkyRL failures, composed from their producers."""

from types import MappingProxyType

from rigging.task_errors import ErrorCategory, compose_error_categories
from shellbox.task_errors import ERROR_CATEGORIES as SHELLBOX_CATEGORIES

TASK_MACHINE_ERROR = "TaskMachineError"
TEMPLATE_ERROR = "TemplateError"
TIMED_OUT_DIAGNOSTIC = "timed_out"


SKYRL_ERROR_CATEGORIES = MappingProxyType(
    {
        "ModelServerError": ErrorCategory.INFRASTRUCTURE,
        "PromptTokenizationError": ErrorCategory.INFRASTRUCTURE,
        TASK_MACHINE_ERROR: ErrorCategory.INFRASTRUCTURE,
        TEMPLATE_ERROR: ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHELLBOX_CATEGORIES, SKYRL_ERROR_CATEGORIES)
