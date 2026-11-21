"""Categories for SkyRL failures, composed from their producers."""

from collections.abc import Mapping
from importlib.util import find_spec
from types import MappingProxyType

from rigging.task_errors import ErrorCategory, compose_error_categories
from shellbox.task_errors import ERROR_CATEGORIES as SHELLBOX_CATEGORIES

_harbor_categories: Mapping[str, ErrorCategory] = MappingProxyType({})
if find_spec("harbor_config") is not None:
    from harbor_config.errors import HARBOR_ERROR_CATEGORIES as _harbor_categories


SKYRL_ERROR_CATEGORIES = MappingProxyType(
    {
        "TaskMachineError": ErrorCategory.INFRASTRUCTURE,
        "TemplateError": ErrorCategory.INFRASTRUCTURE,
    }
)

ERROR_CATEGORIES = compose_error_categories(SHELLBOX_CATEGORIES, _harbor_categories, SKYRL_ERROR_CATEGORIES)
