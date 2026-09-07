from groundguard_rag.domain.enums import RunMode, VerificationState


def test_exactly_five_states():
    # Guards against the states silently being collapsed to a 3-way
    # (supported/contradicted/baseless) classification per project_rules.md.
    assert len(VerificationState) == 5


def test_state_values_match_spec():
    assert VerificationState.SUPPORTED.value == "SUPPORTED"
    assert VerificationState.CONTRADICTED.value == "CONTRADICTED"
    assert VerificationState.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    assert VerificationState.CONFLICTING_EVIDENCE.value == "CONFLICTING_EVIDENCE"
    assert VerificationState.NOT_CHECKABLE.value == "NOT_CHECKABLE"


def test_states_are_distinct_members():
    assert len({state for state in VerificationState}) == 5


def test_run_mode_has_exactly_verify_and_heal():
    assert {member.value for member in RunMode} == {"VERIFY", "HEAL"}
