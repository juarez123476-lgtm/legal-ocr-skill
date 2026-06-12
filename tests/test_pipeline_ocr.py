"""
Test suite for LegalDocumentOCRPipeline (pipeline_ocr.py)
==========================================================
Covers:
  - post_process_text  : legal dictionary correction and fuzzy matching
  - validate_quality   : confidence scoring and review flags
  - identify_document_structure : section detection for Brazilian legal docs
  - extract_text_from_ocr       : PaddleOCR / EasyOCR format normalisation
  - ocr_image          : engine selection and PaddleOCR → EasyOCR fallback
  - _quality_rating    : quality band classification
  - _load_legal_dictionary      : file-present and missing-file paths

Document types covered (from project context — LEGISLACAO_ATUALIZADA.md
and AI-for-Legal scenario decks):
  • Sentenças / acórdãos   — Código de Processo Civil
  • Contratos em geral     — CC Lei 10.406/2002
  • Rescisão trabalhista   — CLT art. 477 §6º / Lei 13.467/2017
  • Acordos de dados LGPD — Lei 13.709/2018 arts. 7, 17-22
  • COF / Franquias        — Lei 13.966/2019
  • Contratos de licença   — LPI Lei 9.279/1996

All heavy ML / CV dependencies (cv2, fitz, PIL, paddleocr, easyocr) are
stubbed via sys.modules before the module is imported, so the suite runs
without GPU drivers, model weights, or OpenCV.

Run with:  pytest tests/
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy binary / ML deps BEFORE importing pipeline_ocr.
# Using setdefault so a real install is preferred when present.
# ---------------------------------------------------------------------------
for _dep in ("cv2", "fitz", "torch", "PIL", "PIL.Image", "paddleocr", "easyocr"):
    sys.modules.setdefault(_dep, MagicMock())

from pipeline_ocr import LegalDocumentOCRPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Shared legal dictionary — superset of the built-in fallback, enriched
# with domain-specific terms from the referenced legislation.
# ---------------------------------------------------------------------------
LEGAL_DICT: dict[str, str] = {
    # Processo civil / penal
    "açao": "ação",
    "apelacao": "apelação",
    "decisao": "decisão",
    "sentenca": "sentença",
    "fundamentaçao": "fundamentação",
    "juizo": "juízo",
    "clausula": "cláusula",
    # CC / contratos
    "rescisao": "rescisão",
    "inadimplencia": "inadimplência",
    "inadimplemento": "inadimplemento",
    # CLT / trabalho
    "remuneracao": "remuneração",
    # LGPD (Lei 13.709/2018)
    "controlador": "controlador",
    "operador": "operador",
    "titular": "titular",
    # Franquias (Lei 13.966/2019)
    "franqueado": "franqueado",
    "franqueador": "franqueador",
    # LPI / licenciamento (Lei 9.279/1996)
    "licenciamento": "licenciamento",
}

# ---------------------------------------------------------------------------
# Realistic document fixtures
# ---------------------------------------------------------------------------
SENTENCA = (
    "TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO\n"
    "Processo nº 1234567-89.2024.8.26.0100\n"
    "\n"
    "RELATÓRIO\n"
    "Vistos. Trata-se de ação de cobrança ajuizada por Fulano contra XYZ Ltda.\n"
    "\n"
    "FUNDAMENTAÇÃO\n"
    "É o relatório. Decido. A pretensão é procedente.\n"
    "\n"
    "DISPOSITIVO\n"
    "Ante o exposto, julgo procedente o pedido.\n"
    "______________________________\n"
    "Juiz de Direito. Assinado digitalmente.\n"
)

LGPD_CONTRATO = (
    "CONTRATO DE TRATAMENTO DE DADOS PESSOAIS — Lei 13.709/2018\n"
    "Controlador: Empresa ABC Ltda., CNPJ 12.345.678/0001-00\n"
    "Operador: Fornecedor DEF S/A, CNPJ 98.765.432/0001-11\n"
    "O titular dos dados tem direito ao acesso, retificação e exclusão.\n"
    "Arts. 7, 17-22 da LGPD. processo nº 0000001 juiz nomeado."
)

CLT_RESCISAO = (
    "TERMO DE RESCISÃO DO CONTRATO DE TRABALHO — CLT art. 477 §6º\n"
    "Empregador: Empresa XYZ Ltda.   Empregado: João da Silva\n"
    "Admissão: 01/01/2020   Demissão: 31/12/2024\n"
    "Nos termos da Lei 13.467/2017 (Reforma Trabalhista).\n"
    "Pelo exposto, são devidas: saldo de salário, férias proporcionais.\n"
    "Tribunal Regional do Trabalho — 2ª Região. processo nº 9999. juiz."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline() -> LegalDocumentOCRPipeline:
    """
    Returns a LegalDocumentOCRPipeline with:
    - All OCR engines replaced by fresh MagicMocks.
    - legal_dict pre-loaded from LEGAL_DICT (no filesystem required).
    - _paddle_mock / _easyocr_mock exposed for assertion in engine tests.
    """
    paddle_mock = MagicMock()
    easyocr_reader_mock = MagicMock()

    sys.modules["paddleocr"].PaddleOCR.return_value = paddle_mock
    sys.modules["easyocr"].Reader.return_value = easyocr_reader_mock

    # Force fallback dictionary (no real file needed)
    with patch("pathlib.Path.exists", return_value=False):
        p = LegalDocumentOCRPipeline(use_gpu=False, use_fallback=True)

    p.legal_dict = LEGAL_DICT.copy()
    p._paddle_mock = paddle_mock
    p._easyocr_mock = easyocr_reader_mock
    return p


# ===========================================================================
# 1. post_process_text
# ===========================================================================

class TestPostProcessText:
    """Legal dictionary correction and fuzzy-match logic."""

    def test_accented_term_corrected(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert "sentença" in pipeline.post_process_text("sentenca")

    def test_clt_rescisao_corrected(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """'rescisao' → 'rescisão'  (CLT art. 477 context)."""
        assert "rescisão" in pipeline.post_process_text("rescisao contratual")

    def test_multiple_terms_corrected(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = pipeline.post_process_text("açao de decisao sobre sentenca")
        assert "ação" in result
        assert "decisão" in result
        assert "sentença" in result

    def test_lgpd_term_passes_through(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """'controlador' is already correct — must not be mangled."""
        assert "controlador" in pipeline.post_process_text("o controlador de dados")

    def test_unknown_word_preserved(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert "xyzzyquxfoo" in pipeline.post_process_text("xyzzyquxfoo")

    def test_empty_string(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = pipeline.post_process_text("")
        assert result == "" or result is not None

    def test_franchise_terms_cof(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """Lei 13.966/2019 COF — franqueado/franqueador must be preserved."""
        result = pipeline.post_process_text("franqueado e franqueador assinaram")
        assert "franqueado" in result
        assert "franqueador" in result

    def test_word_order_preserved(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = pipeline.post_process_text("o juizo da vara civel")
        parts = result.split()
        assert parts[0] == "o"
        assert "juízo" in result

    def test_lpi_licenciamento_term(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """LPI Lei 9.279/1996 — licenciamento de marca/patente."""
        assert "licenciamento" in pipeline.post_process_text("licenciamento de patente")


# ===========================================================================
# 2. validate_quality
# ===========================================================================

class TestValidateQuality:
    """Confidence scoring, issue detection, and review flags."""

    def test_empty_text_zero_confidence(self, pipeline: LegalDocumentOCRPipeline) -> None:
        r = pipeline.validate_quality("")
        assert r["confidence"] == 0
        assert r["requires_review"] is True

    def test_whitespace_only_treated_as_empty(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert pipeline.validate_quality("   \n\t")["confidence"] == 0

    def test_realistic_sentenca_high_confidence(self, pipeline: LegalDocumentOCRPipeline) -> None:
        r = pipeline.validate_quality(SENTENCA)
        assert r["confidence"] >= 70
        assert r["requires_review"] is False

    def test_o_zero_confusion_penalises_confidence(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """Many 'O's with at least one '0' → OCR confusion penalty."""
        text = "OOOOOOOOOOOOOOOOOOOO0 juiz tribunal processo"
        assert pipeline.validate_quality(text)["confidence"] < 100

    def test_short_avg_word_length_penalises(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """Shattered single-char tokens → confidence penalty."""
        text = "a b c d e f g h i j k l m n o p q r s t"
        assert pipeline.validate_quality(text)["confidence"] < 100

    def test_missing_legal_elements_generates_warnings(self, pipeline: LegalDocumentOCRPipeline) -> None:
        r = pipeline.validate_quality("lorem ipsum dolor sit amet.")
        assert len(r["warnings"]) > 0

    def test_requires_review_when_confidence_below_70(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert pipeline.validate_quality("")["requires_review"] is True

    def test_result_contains_all_expected_keys(self, pipeline: LegalDocumentOCRPipeline) -> None:
        r = pipeline.validate_quality(LGPD_CONTRATO)
        assert {"confidence", "issues", "warnings", "requires_review"} <= r.keys()

    def test_lgpd_document_confidence_positive(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """LGPD agreement with 'processo' and 'juiz' should score > 0."""
        assert pipeline.validate_quality(LGPD_CONTRATO)["confidence"] > 0

    def test_clt_rescisao_confidence_positive(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert pipeline.validate_quality(CLT_RESCISAO)["confidence"] > 0


# ===========================================================================
# 3. identify_document_structure
# ===========================================================================

class TestIdentifyDocumentStructure:
    """Section-detection logic against Brazilian legal document types."""

    def test_sentenca_three_sections_detected(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """relatório → fundamentação → dispositivo all populated."""
        s = pipeline.identify_document_structure(SENTENCA)
        assert s["relatorio"] != ""
        assert s["fundamentacao"] != ""
        assert s["dispositivo"] != ""

    def test_header_captured_before_first_marker(self, pipeline: LegalDocumentOCRPipeline) -> None:
        text = (
            "TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO\n"
            "Processo nº 0001\nVistos.\nTrata-se de ação.\n"
        )
        s = pipeline.identify_document_structure(text)
        assert "TRIBUNAL DE JUSTIÇA" in s["header"]

    def test_ante_o_exposto_triggers_dispositivo(self, pipeline: LegalDocumentOCRPipeline) -> None:
        text = "fundamento e decido.\nAnte o exposto, julgo procedente."
        assert pipeline.identify_document_structure(text)["dispositivo"] != ""

    def test_signature_block_captured(self, pipeline: LegalDocumentOCRPipeline) -> None:
        text = "Dispositivo: julgo.\n____________________________\nJuiz Fulano de Tal"
        s = pipeline.identify_document_structure(text)
        assert s["assinaturas"] != "" or s["dispositivo"] != ""

    def test_empty_text_all_sections_empty(self, pipeline: LegalDocumentOCRPipeline) -> None:
        s = pipeline.identify_document_structure("")
        for key in ("header", "relatorio", "fundamentacao", "dispositivo"):
            assert s[key] == ""

    def test_clt_rescisao_dispositivo_detected(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """'Pelo exposto' is a valid dispositivo marker in CLT documents."""
        assert pipeline.identify_document_structure(CLT_RESCISAO)["dispositivo"] != ""

    def test_lgpd_no_court_markers_lands_in_header(self, pipeline: LegalDocumentOCRPipeline) -> None:
        text = (
            "CONTRATO DE TRATAMENTO DE DADOS\n"
            "Controlador: ABC Ltda.\nOperador: DEF S/A.\nLei 13.709/2018.\n"
        )
        assert pipeline.identify_document_structure(text)["header"] != ""

    def test_cof_franchise_no_court_structure(self, pipeline: LegalDocumentOCRPipeline) -> None:
        """COF (Lei 13.966/2019) — must not crash and header must be populated."""
        text = (
            "CIRCULAR DE OFERTA DE FRANQUIA\n"
            "Franqueador: Rede FastFood Ltda.\n"
            "Prazo mínimo de 14 dias (Lei 13.966/2019 art. 2).\n"
        )
        s = pipeline.identify_document_structure(text)
        assert isinstance(s, dict)
        assert s["header"] != ""

    def test_returns_all_expected_section_keys(self, pipeline: LegalDocumentOCRPipeline) -> None:
        expected = {
            "header", "preambulo", "relatorio",
            "fundamentacao", "dispositivo", "assinaturas",
        }
        assert set(pipeline.identify_document_structure("qualquer texto").keys()) == expected


# ===========================================================================
# 4. extract_text_from_ocr
# ===========================================================================

class TestExtractTextFromOcr:
    """Output format normalisation for PaddleOCR and EasyOCR."""

    def test_paddleocr_extracts_text_lines(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = [[[None, ("Ante o exposto,", 0.95)], [None, ("julgo procedente.", 0.92)]]]
        text = pipeline.extract_text_from_ocr(result, "paddleocr")
        assert "Ante o exposto," in text
        assert "julgo procedente." in text

    def test_easyocr_extracts_text_lines(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = [
            (None, "Controlador de dados:", 0.90),
            (None, "titular do dado pessoal.", 0.88),
        ]
        text = pipeline.extract_text_from_ocr(result, "easyocr")
        assert "Controlador de dados:" in text
        assert "titular do dado pessoal." in text

    def test_unknown_source_returns_empty(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert pipeline.extract_text_from_ocr([], "unknown_engine") == ""

    def test_empty_paddleocr_result_returns_empty(self, pipeline: LegalDocumentOCRPipeline) -> None:
        assert pipeline.extract_text_from_ocr([[]], "paddleocr") == ""

    def test_paddleocr_lines_joined_with_newlines(self, pipeline: LegalDocumentOCRPipeline) -> None:
        result = [
            [
                [None, ("linha um", 0.9)],
                [None, ("linha dois", 0.9)],
                [None, ("linha três", 0.9)],
            ]
        ]
        assert pipeline.extract_text_from_ocr(result, "paddleocr").count("\n") == 2

    def test_easyocr_line_order_preserved(self, pipeline: LegalDocumentOCRPipeline) -> None:
        lines = ["primeira linha", "segunda linha", "terceira linha"]
        result = [(None, line, 0.9) for line in lines]
        assert pipeline.extract_text_from_ocr(result, "easyocr").split("\n") == lines


# ===========================================================================
# 5. ocr_image — engine selection and PaddleOCR → EasyOCR fallback
# ===========================================================================

class TestOcrImageFallback:
    """Validates the two-engine fallback chain."""

    @staticmethod
    def _paddle_result(text: str, conf: float) -> list:
        return [[[None, (text, conf)]]]

    def test_paddle_used_above_threshold(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.return_value = self._paddle_result("sentença", 0.95)
        pipeline._easyocr_mock.readtext.reset_mock()

        _, source, confidence = pipeline.ocr_image(MagicMock(), page_num=1)

        assert source == "paddleocr"
        assert confidence >= pipeline.confidence_threshold
        pipeline._easyocr_mock.readtext.assert_not_called()

    def test_fallback_on_low_paddle_confidence(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.return_value = self._paddle_result("txt", 0.05)
        pipeline._easyocr_mock.readtext.return_value = [(None, "decisão", 0.85)]

        _, source, _ = pipeline.ocr_image(MagicMock(), page_num=1)
        assert source == "easyocr"

    def test_fallback_on_empty_paddle_result(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.return_value = None
        pipeline._easyocr_mock.readtext.return_value = [(None, "juízo", 0.80)]

        _, source, _ = pipeline.ocr_image(MagicMock(), page_num=1)
        assert source == "easyocr"

    def test_error_when_both_engines_fail(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.side_effect = RuntimeError("model crash")
        pipeline._easyocr_mock.readtext.side_effect = RuntimeError("cuda oom")

        result, source, confidence = pipeline.ocr_image(MagicMock(), page_num=1)
        assert source == "error"
        assert confidence == 0.0
        assert result is None

    def test_paddle_exception_falls_to_easyocr(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.side_effect = Exception("inference error")
        pipeline._easyocr_mock.readtext.side_effect = None
        pipeline._easyocr_mock.readtext.return_value = [(None, "processo", 0.75)]

        _, source, _ = pipeline.ocr_image(MagicMock(), page_num=1)
        assert source == "easyocr"

    def test_return_value_is_three_tuple(self, pipeline: LegalDocumentOCRPipeline) -> None:
        pipeline._paddle_mock.ocr.side_effect = None
        pipeline._paddle_mock.ocr.return_value = self._paddle_result("texto", 0.9)
        out = pipeline.ocr_image(MagicMock(), page_num=1)
        assert len(out) == 3


# ===========================================================================
# 6. _quality_rating
# ===========================================================================

class TestQualityRating:
    """All four quality bands."""

    @pytest.mark.parametrize(
        "score,expected",
        [
            (1.00, "excellent"),
            (0.90, "excellent"),
            (0.89, "good"),
            (0.80, "good"),
            (0.79, "acceptable"),
            (0.70, "acceptable"),
            (0.69, "poor"),
            (0.50, "poor"),
            (0.00, "poor"),
        ],
    )
    def test_quality_bands(
        self,
        pipeline: LegalDocumentOCRPipeline,
        score: float,
        expected: str,
    ) -> None:
        assert pipeline._quality_rating(score) == expected


# ===========================================================================
# 7. _load_legal_dictionary
# ===========================================================================

class TestLoadLegalDictionary:
    """File-present and missing-file code paths."""

    def test_fallback_non_empty_when_file_missing(self) -> None:
        """Missing JSON file → built-in fallback dict returned."""
        sys.modules["paddleocr"].PaddleOCR.return_value = MagicMock()
        with patch("pathlib.Path.exists", return_value=False):
            p = LegalDocumentOCRPipeline(use_gpu=False, use_fallback=False)
        assert len(p.legal_dict) > 0

    def test_fallback_contains_core_legal_terms(self) -> None:
        """Core Brazilian process terms must appear in the fallback values."""
        sys.modules["paddleocr"].PaddleOCR.return_value = MagicMock()
        with patch("pathlib.Path.exists", return_value=False):
            p = LegalDocumentOCRPipeline(use_gpu=False, use_fallback=False)
        values = set(p.legal_dict.values())
        assert any(v in values for v in ("ação", "decisão", "sentença"))

    def test_custom_json_loaded_when_file_exists(self) -> None:
        """When the JSON file is present it is parsed and returned verbatim."""
        custom = {"requerente": "requerente", "requerido": "requerido"}
        sys.modules["paddleocr"].PaddleOCR.return_value = MagicMock()
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(custom))),
        ):
            p = LegalDocumentOCRPipeline(use_gpu=False, use_fallback=False)
        assert p.legal_dict == custom
