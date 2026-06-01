from __future__ import annotations

import json
import os
import tempfile
import unittest

from mobility_agent.rag.wiki_sync import clean_wikitext, infer_stage, infer_tags, load_house_policy_documents, normalize_title


class WikiSyncTests(unittest.TestCase):
    def test_clean_wikitext_strips_mediawiki_markup(self) -> None:
        raw = """
'''ENCUT''' controls the basis size.<ref name="a">Hidden reference</ref>
Use [[ISMEAR|ISMEAR]] with [https://www.vasp.at/wiki/ENCUT the VASP Wiki].
{{TAGDEF|ENCUT|Plane-wave cutoff energy}}
"""
        cleaned = clean_wikitext(raw)
        self.assertIn("ENCUT", cleaned)
        self.assertIn("ISMEAR", cleaned)
        self.assertIn("the VASP Wiki", cleaned)
        self.assertNotIn("<ref", cleaned)
        self.assertNotIn("TAGDEF", cleaned)
        self.assertNotIn("[[", cleaned)

    def test_stage_and_tag_inference_use_vasp_keywords(self) -> None:
        text = "Electronic minimization with ISMEAR and SIGMA improves self-consistent SCF convergence."
        self.assertEqual(normalize_title("Band structure"), "Band_structure")
        self.assertIn("scf", infer_tags("ISMEAR", text))
        self.assertEqual(infer_stage("ISMEAR", text), "scf")

    def test_load_house_policy_documents_normalizes_json_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "house_policy.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {
                            "corpus": "house_policy",
                            "source_id": "house:encut",
                            "title": "ENCUT Default",
                            "heading": "ENCUT",
                            "stage": "scf",
                            "tags": ["scf", "encut"],
                            "url_or_path": "/tmp/house_policy.json",
                            "text": "Set ENCUT high enough for reliable convergence.",
                        }
                    ],
                    handle,
                )

            documents = load_house_policy_documents(path)

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document.corpus, "house_policy")
        self.assertEqual(document.source_id, "house:encut")
        self.assertEqual(document.title, "ENCUT Default")
        self.assertEqual(document.heading, "ENCUT")
        self.assertEqual(document.stage, "scf")
        self.assertEqual(document.tags, ["scf", "encut"])
        self.assertEqual(document.url, "/tmp/house_policy.json")
        self.assertTrue(document.content_sha)


if __name__ == "__main__":
    unittest.main()
