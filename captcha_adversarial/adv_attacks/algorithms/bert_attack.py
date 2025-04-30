#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bert Attack algorithm implementation.

Refactored based on https://github.com/LinyangLee/BERT-Attack/blob/master/bertattack.py,
with object-oriented encapsulation and added support for Chinese.
"""
import os
import jieba
import copy
import torch
import torch.nn as nn

from PIL import Image
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
from sentence_transformers import SentenceTransformer

__dir__ = os.path.dirname(os.path.abspath(__file__))
jieba.set_dictionary(os.path.join(__dir__, "jieba_dict.txt.big"))

# Stop words list, used to filter out words that should not be replaced during attack
FILTER_WORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "ain",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "am",
    "among",
    "amongst",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "aren",
    "aren't",
    "around",
    "as",
    "at",
    "back",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn",
    "couldn't",
    "d",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "don",
    "don't",
    "down",
    "due",
    "during",
    "either",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "even",
    "ever",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "first",
    "for",
    "former",
    "formerly",
    "from",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "if",
    "in",
    "indeed",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "latter",
    "latterly",
    "least",
    "ll",
    "may",
    "me",
    "meanwhile",
    "mightn",
    "mightn't",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "must",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "namely",
    "needn",
    "needn't",
    "neither",
    "never",
    "nevertheless",
    "next",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "o",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "per",
    "please",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should've",
    "shouldn",
    "shouldn't",
    "somehow",
    "something",
    "sometime",
    "somewhere",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "this",
    "those",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "too",
    "toward",
    "towards",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "used",
    "ve",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "with",
    "within",
    "without",
    "won",
    "won't",
    "would",
    "wouldn",
    "wouldn't",
    "y",
    "yet",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "的",
    "了",
    "是",
    "在",
    "和",
    "有",
    "及",
    "与",
    "对",
    "关于",
    "对于",
    "其",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "这",
    "那",
    "哪",
    "什么",
    "哪里",
    "如何",
    "为什么",
    "是否",
    "之",
    "地",
    "着",
    "过",
    "而",
    "并",
    "或",
    "但",
    "也",
    "都",
    "却",
    "虽然",
    "但是",
    "然而",
    "因此",
    "所以",
    "因为",
    "没有",
    "不是",
    "可以",
    "应该",
    "将会",
    "必须",
    "这个",
    "那个",
    "那些",
    "这里",
    "那里",
    "各种",
    "一些",
    "很多",
    "几个",
    "一种",
    "方面",
    "等等",
}


def split_words(seq):
    """Segment a text string into a list of words.

    Args:
        seq (str): Input text.

    Returns:
        List[str]: List of segmented words.
    """
    return [word.strip() for word in jieba.cut(seq) if word.strip()]


def join_words(words):
    """Join a list of words into a single string.

    Args:
        words (List[str]): List of words.

    Returns:
        str: Joined string.
    """
    text = ""
    for word in words:
        if text and word and "\u4e00" < text[-1] < "\u9fff" and "\u4e00" < word < "\u9fff":
            text += word
        else:
            text += " " + word
    text = text.strip()
    return text


class BertCandidateGenerator(object):
    """BERT-based candidate word generator."""

    def __init__(
        self,
        max_length=512,
        top_k=10,
        candidate_threshold=0.3,
        model_id="google-bert/bert-base-multilingual-cased",
    ):
        """Initialize BertCandidateGenerator.

        Args:
            max_length (int): Maximum sequence length.
            top_k (int): Top-k candidates to consider.
            candidate_threshold (float): Score threshold for candidate filtering.
            model_id (str): Huggingface model identifier.
        """
        print(f"Init BertCandidateGenerator ...")
        self.max_length = max_length
        self.top_k = top_k
        self.candidate_threshold = candidate_threshold
        self.model_id = model_id
        self.tokenizer = BertTokenizer.from_pretrained(model_id, device_map="auto", do_lower_case=True)
        self.model_config = BertConfig.from_pretrained(model_id, device_map="auto")
        self.model = BertForMaskedLM.from_pretrained(model_id, device_map="auto", config=self.model_config)
        self.model.eval()
        self.device = self.model.device

    def prepare_tokens(self, words):
        """Prepare tokens and word-token indices for a list of words.

        Args:
            words (List[str]): List of words.

        Returns:
            Tuple[List[str], List[List[int]]]: Tokens and word-token indices.
        """
        tokens = []
        indexes = []
        for word in words:
            sub_tokens = self.tokenizer.tokenize(word)
            if len(sub_tokens) + len(tokens) > self.max_length - 2:
                break
            indexes.append([len(tokens), len(tokens) + len(sub_tokens)])
            tokens += sub_tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        return tokens, indexes

    def prepare_candidates(self, tokens):
        """Prepare candidate token ids and scores for a list of tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            Tuple[Tensor, Tensor]: Candidate token ids and scores.
        """
        with torch.no_grad():
            input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)], device=self.device)
            outputs = self.model(input_ids)
            scores, token_ids = torch.topk(outputs[0].squeeze(), self.top_k, -1)
            token_ids = token_ids[1 : len(tokens) + 1, :]
            scores = scores[1 : len(tokens) + 1, :]
            return token_ids, scores

    def cross_join_tokens(self, token_ids):
        """Generate all possible combinations of token ids for multi-subword words.

        Args:
            token_ids (Tensor): Token ids.

        Returns:
            List[List[int]]: All possible combinations.
        """
        crossed_ids = []
        for i in range(token_ids.size(0)):
            if len(crossed_ids) == 0:
                lev_i = token_ids[i]
                crossed_ids = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_ids in crossed_ids:
                    for j in token_ids[i]:
                        lev_i.append(all_ids + [int(j)])
                crossed_ids = lev_i
        return crossed_ids

    def get_candidate_words_bpe(self, candidate_token_ids):
        """Generate candidate words for multi-subword tokens using BPE.

        Args:
            candidate_token_ids (Tensor): Candidate token ids.

        Returns:
            List[str]: List of candidate words.
        """
        crossed_token_ids = self.cross_join_tokens(candidate_token_ids[0:12, 0:4])
        crossed_token_ids = torch.tensor(crossed_token_ids)
        crossed_token_ids = crossed_token_ids[:24].to(self.device)  # Limit max length to 24
        with torch.no_grad():
            predict_words = self.model(crossed_token_ids)[0]  # N L vocab-size
            N, L = crossed_token_ids.size()
            CEL = nn.CrossEntropyLoss(reduction="none")
            ppl = CEL(predict_words.view(N * L, -1), crossed_token_ids.view(-1))
            ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
            _, sorted_index = torch.sort(ppl)
            condidate_word_token_ids = [crossed_token_ids[i] for i in sorted_index]
            condidate_words = []
            for word_token_ids in condidate_word_token_ids:
                word_tokens = [self.tokenizer._convert_id_to_token(int(i)) for i in word_token_ids]
                word_text = self.tokenizer.convert_tokens_to_string(word_tokens)
                condidate_words.append(word_text)
            return condidate_words

    def get_candidate_words(self, candidate_token_ids, candidate_scores):
        """Get candidate replacement words predicted by MLM.

        Args:
            candidate_token_ids (Tensor): Candidate token ids from MLM output.
            candidate_scores (Tensor): Candidate word scores.

        Returns:
            List[str]: List of candidate words.
        """
        token_length, _ = candidate_token_ids.size()
        if token_length == 0:
            return []
        elif token_length == 1:
            return [
                self.tokenizer._convert_id_to_token(int(token_id))
                for token_id, score in zip(candidate_token_ids[0], candidate_scores[0])
                if not (self.candidate_threshold != 0 and score < self.candidate_threshold)
            ]
        else:
            return self.get_candidate_words_bpe(candidate_token_ids)

    def generate_candidates(
        self, target_index, target_seq_words, canditate_token_ids, canditate_scores, word_token_indexes
    ):
        """Generate candidate words for a target word in a sentence.

        Args:
            target_index (int): Index of the target word.
            target_seq_words (List[str]): Sentence containing the target word.
            canditate_token_ids (Tensor): Candidate token ids.
            canditate_scores (Tensor): Candidate scores.
            word_token_indexes (List[List[int]]): Word-token indices.

        Yields:
            Tuple[List[str], str]: (Candidate word list, replaced sentence).
        """
        target_word = target_seq_words[target_index]
        if target_word not in FILTER_WORDS:
            target_candidate_token_ids = canditate_token_ids[
                word_token_indexes[target_index][0] : word_token_indexes[target_index][1]
            ]
            target_candidate_scores = canditate_scores[
                word_token_indexes[target_index][0] : word_token_indexes[target_index][1]
            ]
            target_candidate_words = self.get_candidate_words(target_candidate_token_ids, target_candidate_scores)
            print(
                f"Ready to generate candidates for word {target_word}@{target_index} with {target_candidate_words} ..."
            )
            for condidate_word in target_candidate_words:
                if condidate_word != target_word and "##" not in condidate_word and condidate_word not in FILTER_WORDS:
                    condidate_seq_words = (
                        target_seq_words[0:target_index] + [condidate_word] + target_seq_words[target_index + 1 :]
                    )
                    yield condidate_seq_words, join_words(condidate_seq_words)

    def __call__(self, seq):
        """Create a BertCandidateInstance for a given sequence.

        Args:
            seq (str): Input sequence.

        Returns:
            BertCandidateInstance: Instance for candidate generation.
        """
        return BertCandidateInstance(self, seq)


class BertCandidateInstance(object):
    """Instance for generating candidate words for a given sequence."""

    def __init__(self, generator, origin_seq):
        """Initialize BertCandidateInstance.

        Args:
            generator (BertCandidateGenerator): Candidate generator.
            origin_seq (str): Original sequence.
        """
        self.generator = generator
        self.origin_seq = origin_seq
        self.words = split_words(self.origin_seq)
        self.tokens, self.word_token_indexes = self.generator.prepare_tokens(self.words)
        self.canditate_token_ids, self.canditate_scores = self.generator.prepare_candidates(self.tokens)

    def __call__(self, target_index, target_seq_words=None):
        """Generate candidates for a target word.

        Args:
            target_index (int): Index of the target word.
            target_seq_words (List[str], optional): Sequence words. Defaults to None.

        Yields:
            Tuple[List[str], str]: (Candidate word list, replaced sentence).
        """
        yield from self.generator.generate_candidates(
            target_index,
            target_seq_words if target_seq_words else self.words,
            self.canditate_token_ids,
            self.canditate_scores,
            self.word_token_indexes,
        )


class BertSeqClassificationPredictor(object):
    """BERT-based text classification predictor."""

    def __init__(self, max_length=512, num_labels=2, model_id="textattack/bert-base-uncased-yelp-polarity"):
        """Initialize BertSeqClassificationPredictor.

        Args:
            max_length (int): Maximum sequence length.
            num_labels (int): Number of labels.
            model_id (str): Huggingface model identifier.
        """
        print(f"Init BertSeqClassificationPredictor ...")
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id, device_map="auto", do_lower_case=True)
        self.model_config = BertConfig.from_pretrained(self.model_id, device_map="auto", num_labels=num_labels)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_id, device_map="auto", config=self.model_config
        )
        self.model.eval()
        self.device = self.model.device
        self.mask_token = self.tokenizer.unk_token

    def __call__(self, seq):
        """Predict the label and probability for a given text.

        Args:
            seq (str): Input text.

        Returns:
            Tuple[Tensor, Tensor]: (Label, probability tensor).
        """
        inputs = self.tokenizer(seq, max_length=self.max_length, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits.squeeze(), dim=-1)
            label = torch.argmax(probs)
            return label, probs


class TargetModelPredictor(object):
    """Wrapper for a target model predictor."""

    def __init__(self, model, *args, **kwargs):
        """Initialize TargetModelPredictor.

        Args:
            model: Target model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def __call__(self, seq):
        """Predict using the target model.

        Args:
            seq (str): Input sequence.

        Returns:
            Prediction result from the model.
        """
        return self.model.predict(seq, *self.args, **self.kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class BasePredictorFactory(object):
    """Base factory for creating predictors."""

    def __call__(self, *args, **kwds):
        """Create a TargetModelPredictor.

        Returns:
            TargetModelPredictor: Predictor instance.
        """
        return TargetModelPredictor(self, *args, **kwds)


def calculate_importance(predictor, seq_words):
    """Calculate importance scores for each word in a sequence.

    Args:
        predictor: Predictor instance.
        seq_words (List[str]): List of words in the text.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (Original label, original probability, importance scores).
    """
    with torch.no_grad():
        seq = join_words(seq_words)
        origin_label, origin_probs = predictor(seq)
        masked_labels = []
        masked_probs = []
        for i in range(len(seq_words)):
            seq = join_words(seq_words[0:i] + [predictor.mask_token] + seq_words[i + 1 :])
            _label, _probs = predictor(seq)
            masked_labels.append(_label)
            masked_probs.append(_probs)
        masked_labels = torch.stack(masked_labels)
        masked_probs = torch.stack(masked_probs)
        important_scores = (
            origin_probs.max()
            - masked_probs[:, origin_label]
            + (masked_labels != origin_label).float()
            * (masked_probs.max(dim=-1)[0] - torch.index_select(origin_probs, 0, masked_labels))
        )
        return origin_label, origin_probs, important_scores


def bert_attack(candidate_generator, target_predictor, origin_seq, similarity_threshold=None, embedding_model=None):
    """Perform BERT-based adversarial attack on a sequence.

    Args:
        candidate_generator (BertCandidateGenerator): Candidate generator.
        target_predictor: Target predictor.
        origin_seq (str): Original sequence.
        similarity_threshold (float, optional): Similarity threshold for filtering. Defaults to None.
        embedding_model (SentenceTransformer, optional): Embedding model for similarity. Defaults to None.

    Returns:
        dict: Attack result containing success flag, text, label, probabilities, and candidates.
    """
    origin_embedding = None
    if embedding_model is not None:
        origin_embedding = embedding_model.encode(origin_seq)
    candidates = candidate_generator(origin_seq)
    origin_label, origin_probs, important_scores = calculate_importance(target_predictor, candidates.words)
    important_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    current_prob = origin_probs[origin_label]
    print(f'Original Seq: "{origin_seq}" label: {origin_label.item()}, prob: {current_prob.item()}')
    attack_seq_list = []
    attack_seq_words = copy.deepcopy(candidates.words)
    for target_index, important_score in important_index:
        for candidate_words, condidate_seq in candidates(target_index, attack_seq_words):
            print(f"Attack With: {condidate_seq}")
            if embedding_model is not None:
                candidate_embedding = embedding_model.encode(condidate_seq)
                similarity = embedding_model.similarity(candidate_embedding, origin_embedding)
            if similarity_threshold is not None and similarity < similarity_threshold:
                print(f"Similarity: {similarity.item()} < {similarity_threshold}, skip")
                continue
            if embedding_model is not None:
                attack_seq_list.append((condidate_seq, similarity))
            else:
                attack_seq_list.append(condidate_seq)
            attack_label, attack_probs = target_predictor(condidate_seq)
            if attack_label != origin_label:
                print(f"Attack Successed! {origin_label.item()} -> {attack_label.item()}")
                return {
                    "success": True,
                    "text": condidate_seq,
                    "label": attack_label,
                    "probs": attack_probs,
                    "candidates": attack_seq_list,
                }
            else:
                attack_prob = attack_probs[attack_label]
                print(
                    f"Attack Result: {origin_label.item()}: {current_prob.item()} -> {attack_label.item()}: {attack_prob.item()}"
                )
                if attack_prob < current_prob:
                    print(f"Attack Imporved! {current_prob.item()} -> {attack_prob.item()}")
                    attack_seq_words = candidate_words
                    current_prob = attack_prob
    print("Attack Failed!")
    return {
        "success": False,
        "text": None,
        "label": origin_label,
        "probs": origin_probs,
        "candidates": attack_seq_list,
    }


class BertAttacker(object):
    """BERT-based adversarial attacker."""

    def __init__(
        self,
        predictor_factory=None,
        candidate_generator=None,
        embedding_model=None,
        embedding_model_id="all-MiniLM-L6-v2",
    ):
        """Initialize BertAttacker.

        Args:
            predictor_factory: Predictor factory.
            candidate_generator: Candidate generator.
            embedding_model: Embedding model.
            embedding_model_id (str): Embedding model identifier.
        """
        print(f"Init BertAttacker ...")
        self.embedding_model = embedding_model or SentenceTransformer(embedding_model_id)
        self.candidate_generator = candidate_generator or BertCandidateGenerator()
        self.predictor_factory = predictor_factory or BertSeqClassificationPredictor

    def __call__(self, text, *ags, similarity_threshold=0.95, **kwargs):
        """Perform attack on input text.

        Args:
            text (str): Input text.
            similarity_threshold (float, optional): Similarity threshold. Defaults to 0.95.

        Returns:
            dict: Attack result.
        """
        predictor = self.predictor_factory(*ags, **kwargs)
        result = bert_attack(
            self.candidate_generator,
            predictor,
            text,
            embedding_model=self.embedding_model,
            similarity_threshold=similarity_threshold,
        )
        result["candidates"] = sorted(((score.item(), text) for text, score in result["candidates"]), reverse=True)
        return result


if __name__ == "__main__":

    from pprint import pprint

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    bert_generator = BertCandidateGenerator()
    bert_predictor = BertSeqClassificationPredictor()
    pprint(
        bert_attack(
            bert_generator,
            bert_predictor,
            "Is there a way to get the current time in microseconds since the epoch?",
            embedding_model=embedding_model,
            similarity_threshold=0.8,
        )
    )
    print("===" * 20)
    pprint(
        bert_attack(
            bert_generator,
            bert_predictor,
            "有没有办法获取自纪元以来的当前时间的微秒数？",
            embedding_model=embedding_model,
            similarity_threshold=0.8,
        )
    )
