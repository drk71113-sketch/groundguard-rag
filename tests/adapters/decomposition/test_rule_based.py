import pytest

from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import ClaimVerdict, VerificationRequest
from groundguard_rag.domain.ports import EvidenceSelector, Verifier


def _texts(claims):
    return [claim.text for claim in claims]


def _assert_exact_source_spans(answer, claims):
    assert all(answer[claim.start_char : claim.end_char] == claim.text for claim in claims)
    assert all(
        previous.end_char <= current.start_char
        for previous, current in zip(claims, claims[1:])
    )


def test_splits_english_sentences_and_preserves_punctuation():
    answer = "Paris is in France. Hello!"
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ["Paris is in France.", "Hello!"]
    assert [(claim.start_char, claim.end_char) for claim in claims] == [(0, 19), (20, 26)]
    _assert_exact_source_spans(answer, claims)


def test_splits_chinese_sentences():
    answer = "巴黎在法国。你好！请继续。"
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ["巴黎在法国。", "你好！", "请继续。"]
    _assert_exact_source_spans(answer, claims)


def test_semicolons_and_line_breaks_are_boundaries():
    answer = "One; two；\nthree\r\nfour"
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ["One;", "two；", "three", "four"]
    _assert_exact_source_spans(answer, claims)


def test_leading_trailing_and_separator_whitespace_is_excluded_from_spans():
    answer = "  Hello!   World.  "
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ["Hello!", "World."]
    assert [(claim.start_char, claim.end_char) for claim in claims] == [(2, 8), (11, 17)]


def test_repeated_punctuation_and_closing_quote_stay_with_preceding_claim():
    answer = 'He asked, "Really?!" Then left.'
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ['He asked, "Really?!"', "Then left."]
    _assert_exact_source_spans(answer, claims)


def test_decimal_domain_and_email_internal_periods_do_not_split():
    answer = "Version 3.14 is at example.com. Email a.b@example.com? Done."
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == [
        "Version 3.14 is at example.com.",
        "Email a.b@example.com?",
        "Done.",
    ]


def test_common_abbreviations_acronyms_and_initials_do_not_split():
    answer = "Dr. A. Smith used e.g. a test. The U.S. team agreed. Fine."
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == [
        "Dr. A. Smith used e.g. a test.",
        "The U.S. team agreed.",
        "Fine.",
    ]


def test_non_factual_material_is_not_filtered():
    answer = "Hello! I think this is nice. Please continue."
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == ["Hello!", "I think this is nice.", "Please continue."]


def test_text_without_terminal_punctuation_becomes_one_claim():
    answer = "Paris is in France"
    claims = RuleBasedClaimDecomposer().decompose(answer)

    assert _texts(claims) == [answer]
    assert claims[0].claim_id == f"claim-0-{len(answer)}"


def test_punctuation_only_material_is_not_silently_dropped():
    claims = RuleBasedClaimDecomposer().decompose("!!!")
    assert _texts(claims) == ["!!!"]


def test_ids_and_spans_are_deterministic():
    answer = "First. Second."
    decomposer = RuleBasedClaimDecomposer()

    first = decomposer.decompose(answer)
    second = decomposer.decompose(answer)

    assert first == second
    assert [claim.claim_id for claim in first] == ["claim-0-6", "claim-7-14"]


@pytest.mark.parametrize("answer", ["", "   ", "\r\n\t", None, 42])
def test_empty_whitespace_or_non_string_input_is_rejected(answer):
    with pytest.raises(DomainValidationError):
        RuleBasedClaimDecomposer().decompose(answer)


class EmptySelector(EvidenceSelector):
    def select(self, claim, chunks):
        return []


class NotCheckableVerifier(Verifier):
    def verify(self, claim, evidence):
        return ClaimVerdict(
            claim=claim,
            state=VerificationState.NOT_CHECKABLE,
            evidence_assessments=(),
        )


def test_integrates_through_claim_decomposer_port_without_service_coupling():
    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=EmptySelector(),
        verifier=NotCheckableVerifier(),
        calibrator=None,
        config=VerifyConfig(),
        model_revision="rule-baseline-v1",
        threshold_version="none-v1",
        clock=lambda: "2026-09-02T00:00:00+00:00",
    )
    request = VerificationRequest(
        request_id="request-1",
        answer="Hello! Please continue.",
        chunks=(),
    )

    report = service.verify(request)

    assert [verdict.claim.text for verdict in report.verdicts] == [
        "Hello!",
        "Please continue.",
    ]
    assert all(
        verdict.state is VerificationState.NOT_CHECKABLE
        for verdict in report.verdicts
    )
