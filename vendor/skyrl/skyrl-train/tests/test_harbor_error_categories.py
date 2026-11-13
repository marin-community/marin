from harbor_config.errors import error_category, known_error_types
from skyrl_train.utils import harbor_errors


def test_harbor_error_categories_match():
    assert {n: c.value for n, c in harbor_errors._ERROR_CATEGORIES.items()} == {
        n: error_category(n).value for n in known_error_types()
    }
