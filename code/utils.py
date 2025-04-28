import json
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
import argparse
import re
from collections import Counter
import string
import evaluate
from nltk import word_tokenize
from rouge import Rouge




# pipe = pipeline("text-classification",model="sileod/deberta-v3-large-tasksource-nli",device=0)

def evaluate_quiz_questions( candidates, targets: list, pipe, bert_scorer=None, bart_scorer=None):
    if len(candidates) == 0:
        candidates = [""] # to avoid empty candidates

    scores = pipe([dict(text=t, text_pair=c) for c in candidates for t in targets], return_all_scores=True)
    eunmerated_candidates = [c.lower() for c in candidates for t in targets]
    enumerated_targets = [t.lower() for c in candidates for t in targets]
    other_metrics, individual_other_metrics = compute_metrics(eunmerated_candidates, enumerated_targets, bert_scorer=bert_scorer, bart_scorer=bart_scorer)
    all_results = []
    for i, score in enumerate(scores):
        for s in score:
            if s['label'] == 'entailment':
                result_obj = {
                    "candidate": eunmerated_candidates[i],
                    "score": s['score']
                }
                break
        for metric in individual_other_metrics:
            result_obj[metric] = individual_other_metrics[metric][i]
        all_results.append(result_obj)
    return all_results

def postprocess_context(c):
    # remove "Extracted Context"
    c = c.replace("Extracted Context :", "")
    c = c.strip("*").strip("#").strip().replace("\n", " ").replace("-", "")
    c = re.sub(r"\(ID \d+\)", "", c)
    return c.strip()


def parse_quiz_questions(r):
    def replacer(match):
        return match.group(0).replace("\n", " ")
    raw_result = r.strip()
    question_stems = re.findall(r'\*\*Question \d+\*\*\n(.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'Q\d+\. (.*?)\n *?[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\*\*Question \d+:\*\* (.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\*\*Question \d+:\*\*\n(.*?)\n\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'Question \d+:\n(.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'Question \d+: (.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'Question:\n(.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\d+\) (.*?)\n[Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\d+\. \*\*Question \d+:\*\* (.*?)\n *\- [Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\d+\. \*\*(.*?)\n *\- [Aa][\)\.)]', raw_result, re.DOTALL)
    if len(question_stems) == 0:
        question_stems = re.findall(r'\d+\.\s(.*?)\n\s*?-?\s?Answer:', raw_result, re.DOTALL)
    question_stems = [stem.strip().strip("*") for stem in question_stems]
    # final_candidates = [{"question": stem} for stem in question_stems]
    return question_stems


def compute_unigram_f1(reference_text, generated_text):
    """
    Compute unigram F1 score between reference and generated texts
    :param reference_text: Reference text
    :param generated_text: Generated text
    :return: F1 score, precision, recall
    """
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Split into unigrams
        unigrams = text.split()
        return unigrams
    # Preprocess the texts
    reference_unigrams = Counter(preprocess_text(reference_text))
    generated_unigrams = Counter(preprocess_text(generated_text))

    # Find the intersection of unigrams
    intersection_unigrams = reference_unigrams & generated_unigrams
    true_positives = sum(intersection_unigrams.values())

    if true_positives == 0:
        return 0., 0., 0.

    # Calculate precision and recall
    precision = true_positives / sum(generated_unigrams.values())
    recall = true_positives / sum(reference_unigrams.values())

    if precision + recall == 0:
        return 0., 0., 0.

    # Compute F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score, precision, recall




# nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    # def remove_articles(text):
    #     return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))
    # return white_space_fix(remove_articles(remove_punc(lower(s))))

# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

# ROUGE1 score definition
def _rouge1_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]

def compute_rougel_score(predictions, ground_truths):
    rougel_scores = []
    for pred, truth in zip(predictions, ground_truths):
        rougel_scores.append(_rougel_score(pred, truth))
    return rougel_scores
    
def calculate_average(scores):
    return 0 if len(scores) == 0 else sum(scores) / len(scores)

def compute_metrics(predictions, ground_truths, bert_scorer=None, bart_scorer=None):
    bleu = evaluate.load("bleu")
    rougel_scores = []
    rouge1_scores = []
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    bleu_scores = []

    lengths = []
    predictions = [normalize_answer(s) for s in predictions]
    ground_truths = [normalize_answer(s) for s in ground_truths]
    for pred, truth in zip(predictions, ground_truths):
        rougel_scores.append(_rougel_score(pred, truth))
        rouge1_scores.append(_rouge1_score(pred, truth))
        try:
            bleu_score = bleu.compute(predictions=[pred], references=[truth])

            bleu1_scores.append(bleu_score["precisions"][0])
            bleu2_scores.append(bleu_score["precisions"][1])
            bleu3_scores.append(bleu_score["precisions"][2])
            bleu4_scores.append(bleu_score["precisions"][3])
            bleu_scores.append(bleu_score["bleu"])
            # if bleu_score["bleu"] == 0:
            #     print("pred", pred)
            #     print("truth", truth)
            #     print("bleu_score", bleu_score)
        except Exception as e:
            print(e)
            bleu1_scores.append(0)
            bleu2_scores.append(0)
            bleu3_scores.append(0)
            bleu4_scores.append(0)
            bleu_scores.append(0)            
        lengths.append(pred.count(' '))


    avg_rougel = calculate_average(rougel_scores)
    avg_rouge1 = calculate_average(rouge1_scores)
    avg_bleu1 = calculate_average(bleu1_scores)
    avg_bleu2 = calculate_average(bleu2_scores)
    avg_bleu3 = calculate_average(bleu3_scores)
    avg_bleu4 = calculate_average(bleu4_scores)
    avg_bleu = calculate_average(bleu_scores)
    avg_length = calculate_average(lengths)


    metrics = {}
    metrics["length"] = avg_length
    metrics["rouge-1"] = avg_rouge1
    metrics["rouge-l"] = avg_rougel
    metrics["bleu"] = avg_bleu
    metrics["bleu-1"] = avg_bleu1
    metrics["bleu-2"] = avg_bleu2
    metrics["bleu-3"] = avg_bleu3
    metrics["bleu-4"] = avg_bleu4

    individual_metrics = {}
    individual_metrics["length"] = lengths
    individual_metrics["rouge-1"] = rouge1_scores
    individual_metrics["rouge-l"] = rougel_scores
    individual_metrics["bleu"] = bleu_scores
    individual_metrics["bleu-1"] = bleu1_scores
    individual_metrics["bleu-2"] = bleu2_scores
    individual_metrics["bleu-3"] = bleu3_scores
    individual_metrics["bleu-4"] = bleu4_scores
    
    

    if bert_scorer is not None:
        bertscore_precisions, bertscore_recalls, bertscore_f1scores = bert_scorer(predictions, ground_truths, lang='en')
        # Convert tensor to float
        bertscore_precisions = bertscore_precisions.tolist()
        bertscore_recalls = bertscore_recalls.tolist()
        bertscore_f1scores = bertscore_f1scores.tolist()
        bert_f1 = sum(bertscore_f1scores) / len(bertscore_f1scores)
        metrics["bertscore"] = bert_f1
        individual_metrics["bertscore"] = bertscore_f1scores
    if bart_scorer is not None:
        # BARTSCore
        bartscores = bart_scorer.score(predictions, ground_truths, batch_size=1)
        avg_bartscore = sum(bartscores) / len(bartscores)
        metrics["bartscore"] = avg_bartscore
        individual_metrics["bartscore"] = bartscores



    return metrics, individual_metrics
    
def post_process_function(predictions, ground_truths):
    # sanity check
    assert len(predictions) == len(ground_truths)
    Q_predictions = []
    Q_truths = []
    def get_Q(text):
        text = text.strip()
        lines = [line for line in text.split('\n') if line]
        q = lines[0]
        q = q.lstrip("")
        q = q.lstrip("Q: ") if q.startswith("Q: ") else q.lstrip("Question: ")
        return q
    for pred in predictions:
        try:
            Q_predictions.append(get_Q(pred))
        except Exception as e:
            Q_predictions.append("")
            
    for truth in ground_truths:
        Q_truths.append(get_Q(truth))
    return Q_predictions, Q_truths

