"""Package-level smoke tests."""


def test_import_herdlab() -> None:
    import herdlab

    assert herdlab.__version__ == "0.1.0"

