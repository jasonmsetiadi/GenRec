import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recommendation.dataset import item2code
from recommendation.models.generation.prefix_tree import (
    build_trie_from_codebook,
    calculate_sid_pos_index,
)
from recommendation.tokenizer import GenerativeTokenizer


class VariableLengthSidTest(unittest.TestCase):
    def test_ragged_codebook_maps_and_decodes_terminal_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codes.npy"
            rows = np.empty(2, dtype=object)
            rows[0] = np.array([1, 2])
            rows[1] = np.array([2])
            np.save(path, rows)

            item_to_code, _ = item2code(path, [4, 4], [0, 4])

        self.assertEqual(item_to_code[1], [2, 7])
        self.assertEqual(item_to_code[2], [3])

        trie = build_trie_from_codebook(item_to_code.values(), eos_token_id=9)
        self.assertEqual(set(trie.allowed_tokens([])), {2, 3})
        self.assertEqual(trie.allowed_tokens([2]), [7])
        self.assertEqual(trie.allowed_tokens([3]), [9])

    def test_eos_terminated_targets_are_padded_and_matched_by_sid(self):
        config = {
            "max_code_len": 2,
            "model_params": {"max_len": 2},
            "token_params": {
                "pad_token_id": 0,
                "mask_token_id": 8,
                "cls_token_id": 9,
                "sep_token_id": 10,
                "eos_token_id": 11,
            },
        }
        tokenizer = GenerativeTokenizer(config, {1: [2, 7], 2: [3]}, "right")
        batch = tokenizer(
            [
                {"history": [1], "target": 0},
                {"history": [1], "target": 1},
            ]
        )

        self.assertEqual(batch["labels"].tolist(), [[2, 7, 11], [3, 11, -100]])
        preds = torch.tensor([[[2, 7, 11], [3, 11, 0]], [[2, 7, 11], [3, 11, 0]]])
        pos_index = calculate_sid_pos_index(preds, batch["labels"], 11, 0, maxk=2)
        self.assertEqual(pos_index.tolist(), [[True, False], [False, True]])


if __name__ == "__main__":
    unittest.main()
