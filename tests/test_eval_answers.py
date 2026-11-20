import unittest

from scripts.eval_answers import _validate_specs, score_answer


class AnswerEvaluationTests(unittest.TestCase):
    def test_scoring_checks_evidence_answer_citations_and_completion(self) -> None:
        spec = {
            "id": "new-doc-01",
            "query": "What is the limit?",
            "expected_sources": ["rule.pdf"],
            "expected_any_sources": ["appendix.pdf", "rule.pdf"],
            "expected_terms": ["10 ppm"],
            "expected_answer_terms": ["10 ppm"],
            "expected_pages": [4],
        }
        chunks = [
            {
                "source": "rule.pdf",
                "page": 4,
                "text": "The limit is 10 ppm.",
                "citation": "[rule.pdf p.4]",
            }
        ]
        answer = {
            "answer": "The limit is 10 ppm [rule.pdf p.4].",
            "unverified_citations": [],
            "generation_complete": True,
            "claim_support_verified": True,
        }

        result = score_answer(spec, chunks, answer)

        self.assertTrue(result["passed"])
        self.assertEqual(result["evidence_terms_missing"], [])
        self.assertEqual(result["answer_terms_missing"], [])

    def test_validation_accepts_new_specs_and_rejects_bad_page_values(self) -> None:
        valid = [
            {
                "id": "fresh-set",
                "query": "Find the new threshold",
                "expected_answer_terms": ["42 units"],
            }
        ]
        self.assertEqual(_validate_specs(valid), valid)

        with self.assertRaisesRegex(ValueError, "expected_pages"):
            _validate_specs(
                [
                    {
                        "id": "bad",
                        "query": "Query",
                        "expected_pages": [0],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
