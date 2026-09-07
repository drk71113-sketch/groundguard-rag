import pytest

from groundguard_rag.domain.exceptions import (
    ConfigurationError,
    DomainValidationError,
    GroundGuardError,
)


def test_domain_validation_error_is_a_groundguard_error():
    assert issubclass(DomainValidationError, GroundGuardError)


def test_configuration_error_is_a_groundguard_error():
    assert issubclass(ConfigurationError, GroundGuardError)


def test_groundguard_error_is_an_exception():
    assert issubclass(GroundGuardError, Exception)


def test_domain_validation_error_is_raisable_and_catchable_as_base():
    with pytest.raises(GroundGuardError):
        raise DomainValidationError("boom")
