"""
HappyALBERT: a wrapper over PyTorch's ALBERT implementation

"""

# disable pylint TODO warning
# pylint: disable=W0511
import re
from transformers import (
    AlbertForMaskedLM,
    AlbertTokenizer,
)

import torch

from happytransformer.happy_transformer import HappyTransformer


class HappyALBERT(HappyTransformer):
    """
    Currently available public methods:
        AlbertForMaskedLM:
            1. predict_mask(text: str, options=None, k=1)
    """

    def __init__(self, model="bert-base-uncased"):
        super().__init__(model, "ALBERT")
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.qa = None  # Question Answering
        self.tokenizer = AlbertTokenizer.from_pretrained(model)
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

    def _get_masked_language_model(self):
        """
        Initializes the BertForMaskedLM transformer
        """
        self.mlm = AlbertForMaskedLM.from_pretrained(self.model)
        self.mlm.eval()

    def predict_next_sentence(
        self, sentence_a, sentence_b, use_probability=False
    ):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.
        :param sentence_a: First sentence
        :param sentence_b: Second sentence to test if it comes after the first
        :param use_probability: Toggle outputting probability instead of boolean
        :return Result of whether sentence B follows sentence A,
                as either a probability or a boolean
        """

        if not self.__is_one_sentence(
            sentence_a
        ) or not self.__is_one_sentence(sentence_b):
            self.logger.error(
                'Each inputted text variable for the "predict_next_sentence" method must contain a single sentence'
            )
            exit()

        if self.nsp is None:
            self._get_next_sentence_prediction()
        connected = sentence_a + " " + sentence_b
        tokenized_text = self._get_tokenized_text(connected)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(
            tokenized_text
        )
        segments_ids = self._get_segment_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = self.nsp(
                tokens_tensor, token_type_ids=segments_tensors
            )[0]

        probabilities = torch.nn.Softmax(dim=1)(predictions)
        # probability that sentence B follows sentence A
        correct_probability = probabilities[0][0].item()

        return (
            correct_probability
            if use_probability
            else correct_probability >= 0.5
        )

    def __is_one_sentence(self, text):
        """
        Used to verify the proper input requirements for sentence_relation.
        The text must contain no more than a single sentence.
        Casual use of punctuation is accepted, such as using multiple exclamation marks.
        :param text: A body of text
        :return: True if the body of text contains a single sentence, else False
        """
        split_text = re.split("[?.!]", text)
        sentence_found = False
        for possible_sentence in split_text:
            for char in possible_sentence:
                if char.isalpha():
                    if sentence_found:
                        return False
                    sentence_found = True
                    break
        return True

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        if self.qa is None:
            self._get_question_answering()
        input_text = (
            self.cls_token
            + " "
            + question
            + " "
            + self.sep_token
            + " "
            + text
            + " "
            + self.sep_token
        )
        input_ids = self.tokenizer.encode(input_text)
        sep_val = self.tokenizer.encode(self.sep_token)[-1]
        token_type_ids = [
            0 if i <= input_ids.index(sep_val) else 1
            for i in range(len(input_ids))
        ]
        token_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([token_type_ids])
        with torch.no_grad():
            start_scores, end_scores = self.qa(
                input_ids=token_tensor, token_type_ids=segment_tensor
            )
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_list = all_tokens[
            torch.argmax(start_scores) : torch.argmax(end_scores) + 1
        ]
        answer = self.tokenizer.convert_tokens_to_string(answer_list)
        answer = answer.replace(" ' ", "' ").replace("' s ", "'s ")
        return answer
