from pathlib import Path
from typing import Callable, Dict, List, Union

import torch
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import spearmanr, pearsonr
from torch import Tensor
from tqdm import tqdm
from datasets import load_dataset

class STSEvaluatorBase:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, encode: Callable[[List[str]], Tensor]) -> float:
        embeddings1 = encode(self.sentences1)
        embeddings2 = encode(self.sentences2)

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_scores = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_scores = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        cosine_pearson = pearsonr(self.scores, cosine_scores)[0] * 100
        cosine_spearman = spearmanr(self.scores, cosine_scores)[0] * 100

        euclidean_pearson = pearsonr(self.scores, euclidean_scores)[0] * 100
        euclidean_spearman = spearmanr(self.scores, euclidean_scores)[0] * 100

        manhattan_pearson = pearsonr(self.scores, manhattan_scores)[0] * 100
        manhattan_spearman = spearmanr(self.scores, manhattan_scores)[0] * 100

        dot_pearson = pearsonr(self.scores, dot_products)[0]*100
        dot_spearman = spearmanr(self.scores, dot_products)[0]*100

        metrics = [cosine_pearson, cosine_spearman, euclidean_pearson, euclidean_spearman,
                   manhattan_pearson, manhattan_spearman, dot_pearson, dot_spearman]
        avg_metrics = np.mean(metrics)
        
        eval_result = {
            "avg": avg_metrics,
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
            "manhattan_pearson": manhattan_pearson,
            "manhattan_spearman": manhattan_spearman,
            "dot_pearson": dot_pearson,
            "dot_spearman": dot_spearman
        }

        return eval_result


class KlueSTSEvaluator(STSEvaluatorBase):
    def __init__(self):
        scores = []
        dataset = load_dataset("klue", "sts", split="validation")
        sentences1 = dataset["sentence1"]
        sentences2 = dataset["sentence2"]
        scores = [sample["labels"]["label"] for sample in dataset]
        super().__init__(sentences1, sentences2, scores)
        
        
class KorSTSEvaluator(STSEvaluatorBase):
    def __init__(self):
        scores = []
        dataset = load_dataset("dkoterwa/kor-sts", split="valid")
        sentences1 = dataset["sentence1"]
        sentences2 = dataset["sentence2"]
        scores = dataset["score"]
        super().__init__(sentences1, sentences2, scores)
        
        
class STSEvaluation:
    def __init__(self):
        self.sts_evaluators = {
            "kluests": KlueSTSEvaluator(),
            "korsts": KorSTSEvaluator()
        }
        self.dev_evaluator = KlueSTSEvaluator()

    @torch.inference_mode()
    def __call__(
        self,
        encode: Callable[[List[str]], Tensor],
        progress_bar: bool = True,
    ) -> Dict[str, float]:

        results = {}
        if progress_bar:
            iterator = tqdm(
                list(self.sts_evaluators.items()),
                dynamic_ncols=True,
                leave=False,
            )
        else:
            iterator = list(self.sts_evaluators.items())

        for name, evaluator in iterator:
            results[name] = evaluator(encode=encode)

        #results["avg"] = sum(results.values()) / len(results)
        return results

    @torch.inference_mode()
    def dev(
        self,
        encode: Callable[[List[str]], Tensor],
    ) -> float:
        return self.dev_evaluator(encode=encode)
