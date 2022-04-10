import unittest

import torch

from joeynmt.data import PlaintextDataset, load_data, make_data_iter


class TestData(unittest.TestCase):

    def setUp(self):
        self.train_path = "test/data/toy/train"
        self.dev_path = "test/data/toy/dev"
        self.test_path = "test/data/toy/test"
        self.levels = ["char", "word"]  # bpe is equivalently processed to word
        self.max_sent_length = 10
        self.seed = 42

        # minimal data config
        self.data_cfg = {
            "task": "MT",
            "train": self.train_path,
            "dev": self.dev_path,
            "src": {"lang": "de", "level": "word", "lowercase": False,
                    "max_length": self.max_sent_length},
            "trg": {"lang": "en", "level": "word", "lowercase": False,
                    "max_length": self.max_sent_length},
        }

    def testIteratorBatchType(self):

        current_cfg = self.data_cfg.copy()

        # load toy data
        # pylint: disable=unused-variable
        src_vocab, trg_vocab, train_data, dev_data, test_data = \
            load_data(current_cfg)
        # pylint: enable=unused-variable

        # make batches by number of sentences
        train_iter = iter(make_data_iter(
            train_data, batch_size=10, batch_type="sentence", shuffle=True,
            seed=self.seed, pad_index=trg_vocab.pad_index,
            device=torch.device("cpu")))
        batch = next(train_iter)

        self.assertEqual(batch.src.shape[0], 10)
        self.assertEqual(batch.trg.shape[0], 10)

        # make batches by number of tokens
        train_iter = iter(make_data_iter(
            train_data, batch_size=100, batch_type="token", shuffle=True,
            seed=self.seed, pad_index=trg_vocab.pad_index,
            device=torch.device("cpu")))
        b1 = next(train_iter)  # skip a batch
        b2 = next(train_iter)  # skip another batch
        batch = next(train_iter)
        #print(b1.src)
        #print(b2.src)
        self.assertEqual(batch.src.shape, (9, 11))
        self.assertLessEqual(batch.ntokens, 64)

    def testDataLoading(self):
        # pylint: disable=too-many-branches
        # test all combinations of configuration settings
        datasets = ["train", "dev"]
        for test_path in [None, self.test_path]:
            for level in self.levels:
                for lowercase in [True, False]:
                    current_cfg = self.data_cfg.copy()
                    current_cfg["src"]["level"] = level
                    current_cfg["trg"]["level"] = level
                    current_cfg["src"]["lowercase"] = lowercase
                    current_cfg["trg"]["lowercase"] = lowercase
                    if test_path is not None:
                        datasets.append("test")
                        current_cfg["test"] = test_path
                    else:
                        if "test" in datasets:
                            datasets.remove("test")

                    # load the data
                    # pylint: disable=unused-variable
                    src_vocab, trg_vocab, train_data, dev_data, test_data = \
                        load_data(current_cfg, datasets=datasets)
                    # pylint: enable=unused-variable

                    self.assertIs(type(train_data), PlaintextDataset)
                    self.assertIs(type(dev_data), PlaintextDataset)
                    if test_path is None:
                        self.assertIsNone(test_data)
                    else:
                        self.assertIs(type(test_data), PlaintextDataset)

                    # check the number of examples loaded
                    # NOTE: since tokenization is applied in batch construction,
                    # we cannot compute the length and therefore cannot filter examples out
                    # based on the length before batch iteration.
                    expected_train_len = 1000
                    expected_testdev_len = 20  # dev and test have the same len
                    self.assertEqual(len(train_data), expected_train_len)
                    self.assertEqual(len(dev_data), expected_testdev_len)
                    if test_path is None:
                        self.assertIsNone(test_data)
                    else:
                        self.assertEqual(len(test_data), expected_testdev_len)

                    # check the segmentation: src and trg attributes are lists
                    train_src, train_trg = train_data[0]
                    dev_src, dev_trg = dev_data[0]
                    self.assertIs(type(train_src), list)
                    self.assertIs(type(train_trg), list)
                    self.assertIs(type(dev_src), list)
                    self.assertIs(type(dev_trg), list)
                    if test_path is not None:
                        test_src, test_trg = test_data[0]
                        self.assertIs(type(test_src), list)
                        self.assertIs(test_trg, None)

                    # check the length filtering of the training examples
                    train_ex = [train_data[i] for i in range(len(train_data))]
                    src_len, trg_len = zip(*[(len(s), len(t)) for s, t in train_ex
                                             if s is not None and t is not None])
                    self.assertFalse(any(sl > self.max_sent_length for sl in src_len))
                    self.assertFalse(any(tl > self.max_sent_length for tl in trg_len))

                    # check the lowercasing
                    if lowercase:
                        self.assertTrue(
                            all(" ".join(ex).lower() == " ".join(ex)
                                for ex in train_data.src))
                        self.assertTrue(
                            all(" ".join(ex).lower() == " ".join(ex)
                                for ex in dev_data.src))
                        self.assertTrue(
                            all(" ".join(ex).lower() == " ".join(ex)
                                for ex in train_data.trg))
                        self.assertTrue(
                            all(" ".join(ex).lower() == " ".join(ex)
                                for ex in dev_data.trg))
                        if test_path is not None:
                            self.assertTrue(
                                all(" ".join(ex).lower() == " ".join(ex)
                                    for ex in test_data.src))

                    # check the first example from the training set
                    expected_srcs = {"char": "Danke.",
                                     "word": "David Gallo: Das ist Bill Lange."
                                             " Ich bin Dave Gallo."}
                    expected_trgs = {"char": "Thank you.",
                                     "word": "David Gallo: This is Bill Lange. "
                                             "I'm Dave Gallo."}
                    if level == "char":
                        if lowercase:
                            comparison_src = list(expected_srcs[level].lower())
                            comparison_trg = list(expected_trgs[level].lower())
                        else:
                            comparison_src = list(expected_srcs[level])
                            comparison_trg = list(expected_trgs[level])
                    else:
                        if lowercase:
                            comparison_src = expected_srcs[level].lower(). \
                                split()
                            comparison_trg = expected_trgs[level].lower(). \
                                split()
                        else:
                            comparison_src = expected_srcs[level].split()
                            comparison_trg = expected_trgs[level].split()
                    self.assertEqual(train_data.src[0], comparison_src)
                    self.assertEqual(train_data.trg[0], comparison_trg)

    def testRandomSubset(self):
        # only a random subset should be selected for training
        current_cfg = self.data_cfg.copy()
        current_cfg["random_train_subset"] = -1

        # load the data
        # pylint: disable=unused-variable
        src_vocab, trg_vocab, train_data, dev_data, test_data = \
            load_data(current_cfg)
        # NOTE: since tokenization is applied in batch construction,
        # we cannot compute the length and therefore cannot filter examples out
        # based on the length before batch iteration.
        self.assertEqual(len(train_data), 1000)

        current_cfg["random_train_subset"] = 100
        src_vocab, trg_vocab, train_data, dev_data, test_data = \
            load_data(current_cfg)
        train_data.sample_random_subset()
        self.assertEqual(len(train_data), 100)

