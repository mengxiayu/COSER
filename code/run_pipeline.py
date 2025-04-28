import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm
from transformers import (
    set_seed,
    pipeline,
)
from openai import OpenAI
from together import Together

from utils import parse_quiz_questions, evaluate_quiz_questions, compute_unigram_f1, calculate_average

bert_scorer = None
bart_scorer = None



def setup_logger(verbose_level: int) -> logging.Logger:
    """Setup basic logger with the given verbose level."""
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    level_mapping = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    logger.setLevel(level_mapping.get(verbose_level, logging.INFO))
    return logger

def try_generate_response(prompt, client, args, logger, max_attempts: int = 3) -> str:
    """Generates a response using generate_response() with retries if the response is empty."""
    response = generate_response(prompt, client, args, logger)
    attempts = 0
    while not response.strip() and attempts < max_attempts - 1:
        time.sleep(1)
        response = generate_response(prompt, client, args, logger)
        attempts += 1
    return response

def generate_response(prompt, client, args, logger) -> str:
    try:
        common_messages = [
            {"role": "system", "content": "You are a helpful assistant in educational domain."},
            {"role": "user", "content": prompt},
        ]
        # Use different keyword for max token depending on model name
        token_arg = "max_completion_tokens" if "gpt" in args.model_name else "max_tokens"
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=common_messages,
            temperature=0.1,
            **{token_arg: args.max_output_length}
        )
        response_message = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in generating response: {e}")
        response_message = ""
    return response_message



def generate_letters(max_letter):
    # Ensure the letter is uppercase
    max_letter = max_letter.upper()
    
    # Check if the input is a valid letter
    if not ('A' <= max_letter <= 'Z'):
        raise ValueError("Please provide a valid uppercase letter between A and Z")
    
    # Generate letters up to the next letter after max_letter
    letters = [chr(i) for i in range(ord('A'), ord(max_letter) + 1)]
    
    # Convert list to natural language format
    if len(letters) > 2:
        return ', '.join(letters[:-1]) + ' and ' + letters[-1] + '.'
    else:
        return ' and '.join(letters) + '.'

def control_context_length(context, max_len):
    words = context.split()
    words = words[:max_len] if max_len < len(words) else words
    return ' '.join(words)

def extract_unique_contexts(text):
    search_string = "Extracted Context:"
    results = []
    start_pos = 0
    
    while True:
        # Find next occurrence
        index = text.find(search_string, start_pos)
        if index == -1:  # No more occurrences found
            break
        # Find the start of the next occurrence (if any)
        next_index = text.find(search_string, index + 1)
        if next_index == -1:  # This is the last occurrence
            results.append(text[index + len(search_string):].strip())
        else:
            # Add text from this occurrence up to the next one
            results.append(text[index + len(search_string):next_index].strip())
        # Move start position to just after current occurrence
        start_pos = index + 1
    return results
def postprocess_context(c):
    # remove "Extracted Context"
    c = c.replace("Extracted Context :", "")
    c = c.strip("*").strip("#").strip().replace("\n", " ").replace("-", "")
    c = re.sub(r"\(ID \d+\)", "", c)
    return c.strip()

def find_start_id_by_context_length(test_case, max_context_length, context_type):
    # context type: "transcript" or "keyframe"
    
    context_id = test_case["hint_based_idx"]
    start_id = context_id
    context_length_before = int(max_context_length * 0.75)
    for i in range(context_id, -1, -1):
        if context_type == "transcript":
            context_length_before -= len(test_case["transcript"][i]["content"].split())
        elif context_type == "keyframe":
            if test_case["transcript"][i]["keyframe_description"] is not None:
                context_length_before -= len(test_case["transcript"][i]["keyframe_description"].split())
        else:
            raise ValueError("Invalid context type")
        if context_length_before <= 0 or i == 0:
            start_id = i
            break
    return start_id



def rule_based_context_from_lecture(test_case, context_choice):
    if context_choice.startswith("RuleT"):
        try:
            num_transcripts = context_choice[len("RuleT"):]
            num_transcripts = int(num_transcripts)
        except:
            raise ValueError("Invalid context choice:", context_choice)
        return rule_based_context_from_transcript(test_case, num_transcripts)
    elif context_choice.startswith("RuleV"):
        try:
            num_keyframes = context_choice[len("RuleV"):]
            num_keyframes = int(num_keyframes)
        except:
            raise ValueError("Invalid context choice:", context_choice)
        return rule_based_context_from_keyframes(test_case, num_keyframes)
    elif context_choice.startswith("RuleMM"):
        try:
            num_transcripts, num_keyframes = context_choice[len("RuleMM"):].split("-") # e.g., RuleMM3-2
            num_transcripts = int(num_transcripts)
            num_keyframes = int(num_keyframes)
        except:
            raise ValueError("Invalid context choice:", context_choice)
        context_from_transcript = rule_based_context_from_transcript(test_case, num_transcripts)
        context_from_keyframes = rule_based_context_from_keyframes(test_case, num_keyframes)
        context = f"From transcripts: {context_from_transcript}\nFrom slides: {context_from_keyframes}"
        return context
    else:
        raise ValueError("Invalid context choice:", context_choice)
        


def rule_based_context_from_transcript(test_case, num_sentences):
    '''
    Rule-based context selection from transcripts.
    First, parse the context_choice to get the number of sentences.
    Then, select the context using the hint_based_idx as the 75%, and the number of sentences as the context length.
    '''
    # Search forward for the context
    cnt = num_sentences
    start_id = test_case["hint_based_idx"]
    for i in range(start_id, -1, -1):
        cnt -= (test_case["transcript"][i]["content"].count(".") + test_case["transcript"][i]["content"].count("?"))
        if cnt <= 0 or i == 0:
            start_id = i
            break

    context = ""
    current_num_sentences = 0
    for idx, transcript in enumerate(test_case["transcript"]):
        if idx < start_id:
            continue
        current_num_sentences += (transcript["content"].count(".") + transcript["content"].count("?"))
        context += f"(ID {idx})   {transcript['content']}\n"
        if current_num_sentences >= num_sentences:
            break
    print("Number of sentences:", context.count(".") + context.count("?"))        
    return context

def rule_based_context_from_keyframes(test_case, num_keyframes):
    '''
    Rule-based context selection from keyframes.
    First, parse the context_choice to get the number of keyframes.
    Then, select the context using the hint_based_idx as the 75%, and the number of keyframes as the context length.
    '''
    # Search forward for start_id 
    start_id = test_case["hint_based_idx"]
    cnt = num_keyframes
    for i in range(start_id, -1, -1):
        if test_case["transcript"][i]["keyframe_description"] is not None:
            cnt -= 1
        if cnt == 0 or i == 0:
            start_id = i
            break

    
    context = ""
    current_num_keyframes = 0
    for idx, transcript in enumerate(test_case["transcript"]):
        if idx < start_id:
            continue
        if transcript["keyframe_description"] is not None:
            current_num_keyframes += 1
            context += f"(ID {idx})   {transcript['keyframe_description']}\n"
        if current_num_keyframes >= num_keyframes:
            break

    return context


def build_extract_transcript_context_prompt(test_case, max_context_length, CoT_reasoning=False):
    '''
    Select context from transcripts.
    '''
    context_id = test_case["hint_based_idx"]
    start_id = find_start_id_by_context_length(test_case, max_context_length, "transcript")
    full_context = ""
    context_length = 0
    for idx, transcript in enumerate(test_case["transcript"]):
        if idx < start_id:
            continue
        context_length += len(transcript["content"].split())
        if context_length > max_context_length:
            break
        full_context += f"(ID {idx})   {transcript['content']}\n"
    context = full_context
    answer = test_case['answer']['option_text']
    prompt = f"""You are tasked with extracting contexts from the given lecture transcript for generating a quiz question. You'll be provided the following information:

- Lecture Transcript: A lengthy text which is the transcript of a lecture.
- Answer: A word, phrase, or sentence that serves as the answer to a potential quiz question.
- Timestamp: A sentence ID to indicate the time associated with the quiz question.

Output one or multiple pieces of text that meet the following requirements:
- Is self-contained and contains all relevant information for ** creating ** a quiz question, where the answer would be \"{answer}\".
- Might be near the timestamp sentence (ID {context_id}).
- Should contain a ** complete ** piece of information or atomic knowledge, including the subjects that pronouns refer to.
- Is concise and only includes the necessary details relevant to the quiz question.
- Is about 3 to 6 sentences.
- Should match the original transcript text exactly. Do not alter the text in any way.

Lecture Transcript:\n{context}

Answer to quiz question:\n{answer}

Timestamp: ID {context_id}

"""
    if CoT_reasoning:
        prompt += "Let's get started! Please think step by step. First, list all the transcripts that are relevant to the answer and timestamp. Second, double check and adjust each context on a sentence-by-sentence basis based on the requirements. You can add or remove sentences. Lastly, list the final extracted context.\n\nOutput format:\nExtracted Context: [final extracted context here]"
    else:
        prompt += "Let's get started! Please respond with the following format: \nExtracted Context: \"[first context excerpt here]\"\nExtracted Context: \"[second context excerpt here, if applicable]\""
    target = test_case['question'] # this is a placeholder as we don't have the oracle for this task
    return prompt, target

def build_direct_extract_context_prompt(test_case, max_context_length):
    '''
    Select context from transcripts.
    '''
    prompt, target = build_extract_transcript_context_prompt(test_case, max_context_length, CoT_reasoning=False)
    return prompt, target

def build_cot_extract_context_prompt(test_case, max_context_length):
    '''
    Select context from transcripts with Chain-of-Thought reasoning.
    '''
    prompt, target = build_extract_transcript_context_prompt(test_case, max_context_length, CoT_reasoning=True)
    return prompt, target
    
def build_extract_vision_context_prompt(test_case, max_context_length, CoT_reasoning=False):
    '''
    Select context from keyframes.
    '''
    context_id = test_case["hint_based_idx"]
    # Search backward for the start_id
    start_id = find_start_id_by_context_length(test_case, max_context_length, "keyframe")
    # Search forward for the context
    full_context = ""
    context_length = 0
    keyframe_start_id = start_id
    keyframe_id = 0
    hint_keyframe_id = 0
    for idx, transcript in enumerate(test_case["transcript"]):
        if idx < start_id:
            continue
        if transcript["keyframe_description"] is not None:
            context_length += len(transcript["keyframe_description"].split())
            if context_length > max_context_length:
                break
            keyframe_id += 1
            full_context += f"(ID {keyframe_id}): {transcript['keyframe_description']}\n"
        if idx == context_id:
            hint_keyframe_id = keyframe_id
    context = full_context
    answer = test_case['answer']['option_text']
    prompt = f"""You are tasked with extracting contexts from the given lecture materials for generating a quiz question. You'll be provided the following information:

- Slide Description: Description of a set of slides of the lecture.
- Answer: A word, phrase, or sentence that serves as the answer to a potential quiz question.
- Timestamp: A slide ID to indicate the time associated with the quiz question.

Output the slide description that meets the following requirements:
- Is of the slides that are most relevant to the quiz question, where the answer would be \"{answer}\".
- Is sufficient and provides all relevant contexts for ** creating ** a quiz question.
- Might be near the slide ID {hint_keyframe_id}.
- Should contain the discriptions of 3 to 5 slides.
- Should match the original slide description text exactly. Do not alter the text in any way.

Slide Description:\n{context}

Answer to quiz question:\n{answer}

Timestamp: ID {hint_keyframe_id}

"""
    if CoT_reasoning:
        prompt += f"\nLet's get started! Please think step by step. First, list all the transcripts that are relevant to the answer and timestamp. Second, double check and adjust each context on a sentence-by-sentence basis based on the requirements. You can add or remove sentences. Lastly, list the final extracted context.\n\nOutput format:\n\nReasoning: [your reasoning process] \nExtracted Context: From Slides: [final extracted slide excerpt]\""
    else:
        prompt += "Let's get started! Please respond with the following format: \nExtracted Context: \"[slides excerpt here]"
    target = test_case['question'] # this is a placeholder as we don't have the oracle for this task
    return prompt, target

def build_direct_extract_vision_context_prompt(test_case, args): 
    '''
    Select context from keyframes.
    '''
    prompt, target = build_extract_vision_context_prompt(test_case, args, CoT_reasoning=False)
    return prompt, target

def build_cot_extract_vision_context_prompt(test_case, args):
    '''
    Select context from keyframes with Chain-of-Thought reasoning.
    '''
    prompt, target = build_extract_vision_context_prompt(test_case, args, CoT_reasoning=True)
    return prompt, target

def build_extract_multimodal_context_prompt(test_case, max_context_length, CoT_reasoning=False):
    
    start_id = find_start_id_by_context_length(test_case, int(max_context_length * 0.6), "transcript")
    context_id = test_case["hint_based_idx"]

    transcript_context = ""
    keyframe_context = ""
    context_length = 0
    previous_keyframe_id = start_id
    previous_keyframe = None
    keyframe_context_id_range = f"(ID {start_id} to ID {start_id})"
    for idx, transcript in enumerate(test_case["transcript"]):
        if idx < start_id:
            continue
        if context_length > max_context_length:
            break

        context_length += len(transcript["content"].split())
        transcript_context += f"(ID {idx})   {transcript['content']}\n"
        if transcript["keyframe_description"] is not None:
            if previous_keyframe is not None:
                keyframe_context += f"(ID {previous_keyframe_id} to ID {idx}): {previous_keyframe}\n"
                context_length += len(previous_keyframe.split())
                if context_id < idx and context_id >= previous_keyframe_id:
                    keyframe_context_id_range = f"(ID {previous_keyframe_id} to ID {idx})"
            previous_keyframe_id = idx
            previous_keyframe = transcript["keyframe_description"].replace("\n", " ")
            
    # add the last keyframe
    if previous_keyframe is not None:
        keyframe_context += f"(ID {previous_keyframe_id} to ID {idx}): {previous_keyframe}\n"
        if context_id >= previous_keyframe_id:
            keyframe_context_id_range = f"ID {previous_keyframe_id} to ID {idx}"
    
            
    answer = test_case['answer']['option_text']
    prompt = f"""You are tasked with extracting contexts from the given lecture materials for generating a quiz question. You'll be provided the following information:

- Lecture Transcript: A lengthy textual transcript of a lecture.
- Slide Description: Description of slides that are associated with the lecture transcript.
- Answer: A word, phrase, or sentence that serves as the answer to a potential quiz question.
- Timestamp: A sentence ID to indicate the time position associated with the quiz question.

You output must include the following three components: 

1. Extracted Transcript Excerpt:
- Select a segment directly from the lecture transcript.
- It must be self-contained, clearly providing all information required for ** creating ** a quiz question with the provided the answer \"{answer}\".
- Might be near the transcript timestamp ID {context_id}.
- Should contain a ** complete ** piece of information or atomic knowledge, clearly specifying subjects referenced by pronouns.
- Is concise yet comprehensive, spanning approximately 5 to 8 sentences.
- Should match the original transcript text exactly. Do not alter the text in any way.

2. Extracted Slide Description Excerpt:
- Select a segment from the slides that are most relevant to the quiz question.
- Provide sufficient context necessary for generating a quiz question, where the answer would be \"{answer}\".
- Might be located near slides identified by {keyframe_context_id_range}.
- Should include descriptions from approximately 1 to 3 slides.
- Should match the original slide description text exactly. Do not alter the text in any way.

3. Contextual Summary:
- Provide a brief and clear summary of the main ideas or topics discussed immediately before the selected transcript excerpt.
- The summary should frame and support the context for the chosen excerpt, enabling a quiz creator to understand the background necessary for formulating the question.

Lecture Transcript:\n{transcript_context}

Slide Description:\n{keyframe_context}

Answer to quiz question:\n{answer}

Timestamp: ID {context_id}

"""
    if CoT_reasoning:
        prompt += f"\nLet's get started! Please think step by step. First, list all the transcripts that are relevant to the answer and timestamp. Second, double check and adjust each context on a sentence-by-sentence basis based on the requirements. You can add or remove sentences. Lastly, list the final extracted context.\n\nOutput format:\n\nReasoning: [your reasoning process] \nExtracted Context: From transcripts: [final extracted context from transcript]\nFrom slides: [context from slide description]\nContextual summary: [contextual summary here]\n"
    else:
        prompt += f"\nLet's get started! Please respond with the following format: \nExtracted Context: \"From transcripts: [context from transcript here]\nFrom slides: [context from slide description here, if applicable.]\"\nContextual Summary: [contextual summary here]"
    target = test_case['question'] # placeholder
    return prompt, target

def build_direct_extract_multimodal_context_prompt(test_case, max_context_length):
    '''
    Select context from transcripts and keyframes.
    '''
    prompt, target = build_extract_multimodal_context_prompt(test_case, max_context_length, CoT_reasoning=False)
    return prompt, target

def build_cot_extract_multimodal_context_prompt(test_case, max_context_length):
    '''
    Select context from transcripts and keyframes with Chain-of-Thought reasoning.
    '''
    prompt, target = build_extract_multimodal_context_prompt(test_case, max_context_length, CoT_reasoning=True)
    return prompt, target

def rewrite_context_prompt(test_case):
    target = test_case['question']
    context = test_case["predicted_context"]
    answer = test_case["answer"]["option_text"]
    prompt = f'''You task is to rewrite the contexts extracted from a lecture for creating quiz questions assessing the understanding of the lecture. You'll be provided the following information:\n\n- Extracted Context: A piece of context extracted from the lecture transcript or slides.\n- Answer: A word, phrase, or sentence that serves as the answer to a potential quiz question.\n\nYou can rewrite the contexts into 3-5 statements. Each statement should meet the following requirements:\n- Most importantly, the provided answer \"{answer}\" must appear WORD-FOR-WORD.\n- Is concise and contains a piece of atomic knowledge or a key concept, which could be an important learning point to be assessed in a quiz.\n- Should preserve the original lecture language. Keep all technical/domain-specific terms exactly as presented. Try NOT to alter the text. Only allows changes for: Grammatical connections, removing reference words (this, that, it, etc.), and converting indirect to direct speech.\n\nNotes:\n- If no statements can be directly rewritten from the contexts with the provided answer, then you may write a statement inferred from the context, as long as it contains the provided answer WORD-FOR-WORD.\nYou don't have to incorporate all of the contents into your rewritten statements. Just focus on those relevant to the provided answer.\n\nContexts:\n{context}\nAnswer:{answer}\n\nLet's get started! Please think step by step. First, list a few statements. Next, check whether they satisfy the above requirements and adjust them accordingly. Finally, output the final rewritten statements. Please respond in this format:\n\n Reasoning: [your reasoning process]\nRewritten Contexts: [list of final rewritten statements here]'''
    return prompt, target

def rewrite_multimodal_context_prompt(test_case):

    target = test_case['oracle-gpt4o'] if 'oracle-gpt4o' in test_case else test_case['question']
    context = test_case["predicted_context"]
    answer = test_case["answer"]["option_text"]
    prompt = f'''You task is to rewrite the contexts extracted from a lecture for creating quiz questions assessing the understanding of the lecture. You'll be provided the following information:\n\n- Extracted Context: A piece of context extracted from the lecture transcript or slides.\n- Answer: A word, phrase, or sentence that serves as the answer to a potential quiz question.\n\nYou can rewrite the transcripts into 3-5 statements. Each statement should meet the following requirements:\n- Most importantly, the provided answer \"{answer}\" must appear WORD-FOR-WORD.\n- Should a piece of knowledge which could be an important learning point to be assessed in a quiz.\n- Should preserve the original lecture language. Keep all technical/domain-specific terms exactly as presented. Try NOT to alter the text. Only allows changes for: Grammatical connections, removing reference words (this, that, it, etc.), and converting indirect to direct speech.\n\nNotes:\n- If no statements can be directly rewritten from the transcripts with the provided answer, then you may write a statement inferred from the context, as long as it contains the provided answer WORD-FOR-WORD.\nThe final statements should be rewritten from the transcripts, but you can integrate the information from the keyframes and contextual summary into it.\nYou don't have to incorporate all the information into rewritten statements. Just focus on the ones that are relevant to the provided answer.\nImportantly, the goal is to capture a structured set of knowledge points of different levels of granularity, ranging from general or conceptual statements to specific or technical details. \nContexts:\n{context}\nAnswer:{answer}\n\nPlease think step by step. Please respond in this format:\n\n Reasoning: [your reasoning process]\nRewritten Contexts: [list of final rewritten statements here]'''
        
    return prompt, target

def ca2q_rewrite_prompt(test_case, context_choice):
    context = test_case['predicted_context']
    prompt = f'''You're an experienced STEM teacher. Your task to rephrase each of the following statements into a multiple-choice quiz question. All questions should naturally lead to the answer \"{test_case['answer']['option_text']}\". The questions should be concise and preserve the original language of the statements, and should be interrogative sentences. You don't have to incorporate all the information from a statement into a question. You can write 5 questions in total.\n\n'''

    prompt += "\n\nUse this format for each question:\nQ1. [Question text]\nA) [Option A]]nB) [Option B]\nC) [Option C]\nD) [Option D]"           
    prompt += f"\n\nLecture Content (a list of statements):\n{context}"
    prompt += f"\n\nCorrect Answer:\n\"{test_case['answer']['option_text']}\""
    prompt += f"\n\nPlease provide your questions below."
    target = test_case['question']  
    return prompt, target

def ca2q_prompt(test_case, context_choice):
    context = test_case['predicted_context']
    prompt = f'''You're an experienced STEM teacher. Your task to generate multiple-choice quiz questions from the lecture content. All questions should naturally lead to the answer \"{test_case['answer']['option_text']}\". The questions should be concise and preserve the original language of the lecture content, and should be interrogative sentences. You don't have to incorporate all the information from the lecture content into the questions. You can write 5 questions in total. \n\n'''

    prompt += "\n\nUse this format for each question:\nQ1. [Question text]\nA) [Option A]]nB) [Option B]\nC) [Option C]\nD) [Option D]"           
    prompt += f"\n\nLecture Content:\n{context}"
    prompt += f"\n\nCorrect Answer:\n\"{test_case['answer']['option_text']}\""
    if "Full" in context_choice or "Rule" in context_choice: # provide timestamp for rule-based context
        prompt += f"\n\nQuestion Timestamp: ID {test_case['hint_based_idx']}"
    prompt += f"\n\nPlease provide your questions below."
    target = test_case['question']  

    return prompt, target




def main(args):
    if 'deepseek' in args.model_name:
        client = OpenAI(
            api_key=args.api_key,
            base_url="https://api.deepseek.com"
        )
    elif "Llama" in args.model_name or "Qwen" in args.model_name:
        client = Together(api_key=args.api_key)
    else:
        client = OpenAI(
        api_key=args.api_key,
        # 
    )
    
    logger = setup_logger(args.verbose)
    # 1. Load data
    set_seed(args.seed)
    with open(args.data_file) as f:
        data = [json.loads(line) for line in f]
        if args.max_test_samples:
            data = data[:args.max_test_samples] if args.max_test_samples < len(data) else data
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)    
    # Save configuration
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(args)
    # Load evaluation model
    pipe = pipeline("text-classification",model="sileod/deberta-v3-large-tasksource-nli", device=args.device)


    PROMPT_BUILDERS = {
        # Context Extraction Prompts
        "DirectT": build_direct_extract_context_prompt,
        "CoTT": build_cot_extract_context_prompt,
        "DirectV": build_direct_extract_vision_context_prompt,
        "CoTV": build_cot_extract_vision_context_prompt,
        "DirectMM": build_direct_extract_multimodal_context_prompt,
        "CoTMM": build_cot_extract_multimodal_context_prompt,
        "RuleT": rule_based_context_from_lecture,
        "RuleV": rule_based_context_from_lecture,
        "RuleMM": rule_based_context_from_lecture,
        # Context Rewriting Prompts
        "rewriteContext": rewrite_context_prompt,
        "rewriteMultiModalContext": rewrite_multimodal_context_prompt,
        # Question Generation Prompts
        "CA2Q": ca2q_prompt,
        "CA2Q_rephrase": ca2q_rewrite_prompt,

    }
    def to_text_prompt(test_case, action):
        if action in PROMPT_BUILDERS:
            if action in ["DirectT", "CoTT", "DirectV", "CoTV", "DirectMM", "CoTMM"]:
                return PROMPT_BUILDERS[action](test_case, args.max_context_length)
            elif action in ["rewriteContext", "rewriteMultiModalContext"]:
                return PROMPT_BUILDERS[action](test_case)
            else:
                # Rule-based context or rewriting context or question generation
                return PROMPT_BUILDERS[action](test_case, args.context_choice)

        else:
            raise ValueError(f"Invalid action: {action}")
    all_conversations = []
    all_extracted_contexts = []
    all_rewritten_contexts = []
    all_predicted_questions = []
    all_target_questions = []
    all_target_contexts = []
    all_candidate_nli_scores = []
    all_max_nli_scores = []
    all_context_f1 = []
    all_context_precision = []
    all_context_recall = []
    all_other_metrics = {
        "length": [],
        "rouge-1": [],
        "rouge-l": [],
        "bleu": [],
        "bleu-1": [],
        "bleu-2": [],
        "bleu-3": [],
        "bleu-4": [],
        "bertscore": [],
        "bartscore": []
    }


    for idx, test_case in tqdm(enumerate(data)):
        conversation = []
        # 1. Extract Context
        if args.context_choice in ["DirectT", "CoTT", "CoTV", "DirectV", "DirectMM", "CoTMM"]:
            prompt, target = to_text_prompt(test_case, args.context_choice)
            response = try_generate_response(prompt, client, args, logger)

            conversation.append([{ "role": "user", "content": prompt }, { "role": "system", "content": response }])
            contexts = extract_unique_contexts(response)
            contexts = [postprocess_context(c) for c in contexts]
            predicted_context = '\n'.join(contexts)
            if predicted_context.strip() == "":
                # if no context is extracted, use rule-based context
                predicted_context = rule_based_context_from_transcript(test_case, 9)
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context

        elif args.context_choice == "CombineMM":
            # First extract context from transcript
            prompt, target = to_text_prompt(test_case, "CoTExtractContext")
            response = try_generate_response(prompt, client, args, logger)
            conversation.append([{ "role": "user", "content": prompt }, { "role": "system", "content": response }])
            contexts = extract_unique_contexts(response)
            contexts = [postprocess_context(c) for c in contexts]
            context_from_transcript = '\n'.join(contexts)   
            if context_from_transcript.strip() == "":
                # if no context is extracted, use rule-based context
                context_from_transcript = rule_based_context_from_transcript(test_case, 9)
            # Then extract context from vision information (i.e., video keyframes)
            prompt, target = to_text_prompt(test_case, "CoTExtractVisionContext")
            response = try_generate_response(prompt, client, args, logger)
            conversation.append([{ "role": "user", "content": prompt }, { "role": "system", "content": response }])
            contexts = extract_unique_contexts(response)
            contexts = [postprocess_context(c) for c in contexts]
            context_from_keyframes = '\n'.join(contexts)   
            if context_from_keyframes.strip() == "":
                # if no context is extracted, use rule-based context
                context_from_keyframes =  rule_based_context_from_keyframes(test_case, 3)
            predicted_context = f"{context_from_transcript}\n{context_from_keyframes}"
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context
                       

        elif args.context_choice == "FullVision":
            # Full context from vision information (i.e., video keyframes)
            context_id = test_case["hint_based_idx"]
            start_id = context_id
            context_length_before = int(args.max_context_length * 0.75)
            keyframe_context = ""
            keyframe_start_id = start_id
            context_length = 0
            for i, transcript in enumerate(test_case["transcript"]):
                if i < start_id:
                    continue
                if context_length > args.max_context_length:
                    break
                if transcript["keyframe_description"] is not None:
                    try:
                        keyframe_discription = transcript["keyframe_description"].replace("\n", " ")
                    except:
                        raise ValueError("Invalid keyframe description:", transcript["keyframe_description"])
                    keyframe_context += f"(ID {keyframe_start_id} to ID {i}): {keyframe_discription}\n"
                    context_length += len(transcript["keyframe_description"].split())
                    keyframe_start_id = i
                else:
                    continue
            predicted_context = keyframe_context
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context

        elif args.context_choice == "Full":
            # use all transcripts
            context_id = test_case["hint_based_idx"]
            start_id = context_id
            context_length_before = int(args.max_context_length * 0.75)
            for i in range(context_id, -1, -1):
                context_length_before -= len(test_case["transcript"][i]["content"].split())
                if context_length_before <= 0 or i == 0:
                    start_id = i
                    break
            full_context = ""
            context_length = 0
            for transcript_idx, transcript in enumerate(test_case["transcript"]):
                if transcript_idx < start_id:
                    continue
                context_length += len(transcript["content"].split())
                if context_length > args.max_context_length:
                    break
                full_context += f"(ID {transcript_idx})   {transcript['content']}\n"
            predicted_context = full_context
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context
        elif args.context_choice == "Oracle":
            predicted_context = test_case["oracle-gpt4o"]
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context     
        elif args.context_choice.startswith("Rule"):
            predicted_context = rule_based_context_from_lecture(test_case, args.context_choice)
            all_extracted_contexts.append(predicted_context)
            test_case["predicted_context"] = predicted_context
            
        else:
            raise ValueError("Invalid context choice")
        
        _f1, _precision, _recall = compute_unigram_f1(
            generated_text=test_case["predicted_context"], 
            reference_text=test_case["oracle-gpt4o"] if "oracle-gpt4o" in test_case else test_case["question"],
            )
        logger.info(f"Extracted context: {test_case['predicted_context']}")
        logger.info(f"Context F1: {_f1:.4f}, Precision: {_precision:.4f}, Recall: {_recall:.4f}")
        all_context_f1.append(_f1)
        all_context_precision.append(_precision)
        all_context_recall.append(_recall)

        # 2. Rewrite Context
        if args.rewrite_choice == "Yes":
            if args.context_choice in ["DirectMM", "CoTMM","CombineMM"]:
                prompt, target = to_text_prompt(test_case, "rewriteMultiModalContext")
            else:
                prompt, target = to_text_prompt(test_case, "rewriteContext")
            
            response = try_generate_response(prompt, client, args, logger)
            if response.strip() == "":
                logger.error(f"Empty response for rewrite context: {idx}")

            # postprocess response
            _lines = response.split("\n")
            final_response = ""
            for line in _lines:
                if "Here are the" in line:
                    continue
                elif line.strip() == "":
                    continue
                else:
                    final_response += line + "\n"
            # remove reasoning process if COT is used for context rewriting
            rewritten_context_index = final_response.find("Rewritten Contexts:")
            if rewritten_context_index != -1:
                final_response = final_response[rewritten_context_index + len("Rewritten Contexts:"):].strip()
            conversation.append([{ "role": "user", "content": prompt }, { "role": "system", "content": response }])
            # update predicted context
            test_case["predicted_context"] = final_response
            all_rewritten_contexts.append(final_response)
        elif args.rewrite_choice == "No":
            all_rewritten_contexts.append("")
        else:
            raise ValueError("Invalid rewrite choice")
        
        # 3. Question Generation
        if args.rewrite_choice == "Yes":
            prompt, target = to_text_prompt(test_case, "CA2Q_rephrase")  
            # prompt, target = to_text_prompt(test_case, "CA2Q")
        else:
            prompt, target = to_text_prompt(test_case, "CA2Q")
        response = try_generate_response(prompt, client, args, logger)
        if response.strip() == "":
            print("Empty response", idx)
        conversation.append([{ "role": "user", "content": prompt }, { "role": "system", "content": response }])
        test_case["predicted_questions"] = response
        logger.info(f"Generated questions: {response}")

        # 4. Evaluate
        predQs = parse_quiz_questions(test_case["predicted_questions"]) if test_case["predicted_questions"] else [""]
        targetQs = [test_case["question"], test_case["rephrased_question"]] if "rephrased_question" in test_case else [test_case["question"]] # support multiple targets
        predQ_w_scores = evaluate_quiz_questions(predQs, targetQs, pipe, bert_scorer, bart_scorer)
        # sort by score
        predQ_w_scores = sorted(predQ_w_scores, key=lambda x: x["score"], reverse=True)


        logger.info(f"Target questions: {targetQs}")
        for s in predQ_w_scores:
            logger.info(f"Candidate: {s['candidate']}")
            logger.info(f"Score: {s['score']:.4f}")
        # max score
        if len(predQ_w_scores) > 0:
            max_score = max([s["score"] for s in predQ_w_scores])
        else:
            max_score = 0
        
        if len(predQ_w_scores) > 0:
            max_score_id = [i for i, s in enumerate(predQ_w_scores) if s["score"] == max_score][0]
            # other metrics
            for metric in all_other_metrics:
                if metric in predQ_w_scores[max_score_id]:
                    all_other_metrics[metric].append(predQ_w_scores[max_score_id][metric])
        else:
            for metric in all_other_metrics:
                all_other_metrics[metric].append(0)
        all_candidate_nli_scores.append(predQ_w_scores)
        all_max_nli_scores.append(max_score)
        all_conversations.append(conversation)
        all_predicted_questions.append(test_case["predicted_questions"])
        all_target_questions.append(targetQs)
        if "oracle-gpt4o" in test_case:
            all_target_contexts.append(test_case["oracle-gpt4o"]) 
        else:
            all_target_contexts.append(test_case["question"])
        try:
            assert len(all_predicted_questions) == len(all_target_questions) == len(all_target_contexts) == len(all_max_nli_scores) == len(all_candidate_nli_scores) == len(all_conversations) == len(all_extracted_contexts) == len(all_rewritten_contexts) == idx + 1
        except AssertionError:
            logger.error(f"Length mismatch. Index: {idx}. all_predicted_questions: {len(all_predicted_questions)}, all_target_questions: {len(all_target_questions)}, all_target_contexts: {len(all_target_contexts)}, all_max_nli_scores: {len(all_max_nli_scores)}, all_candidate_nli_scores: {len(all_candidate_nli_scores)}, all_conversations: {len(all_conversations)}, all_extracted_contexts: {len(all_extracted_contexts)}")
            sys.exit(1)
        # user logger
        logger.info(f"Index: {idx}, Max NLI Score: {max_score:.4f}")
    # print results 4 digits after the decimal point
    print(f"Average context F1: {sum(all_context_f1) / len(all_context_f1):.4f}")
    print(f"Average context Precision: {sum(all_context_precision) / len(all_context_precision):.4f}")
    print(f"Average context Recall: {sum(all_context_recall) / len(all_context_recall):.4f}")
    print(f"Average max NLI score: {sum(all_max_nli_scores) / len(all_max_nli_scores):.4f}")
    # print other metrics
    for metric in all_other_metrics:
        if len(all_other_metrics[metric]) > 0:
            avg_score = calculate_average(all_other_metrics[metric])
            print(f"Average {metric}: {avg_score:.4f}")


    # 5. Save results
    with open(output_dir / "result.json", 'w') as f:
        for idx, test_case in enumerate(data):
            r = {
                "idx": idx,
                "extracted_context": all_extracted_contexts[idx],
                "rewritten_context": all_rewritten_contexts[idx],
                "predicted_questions": all_predicted_questions[idx],
                "target_questions": all_target_questions[idx],
                "target_context": all_target_contexts[idx],
                "max_nli_score": all_max_nli_scores[idx],
                "candidate_nli_scores": all_candidate_nli_scores[idx],
                "conversation": all_conversations[idx]
            }
            f.write(json.dumps(r) + "\n")
    with open(output_dir / "qg_result.json", 'w') as f:
        for idx, test_case in enumerate(data):
            r = {
                "raw_result": all_predicted_questions[idx],
                "target": all_target_questions[idx],
                "max_nli_score": all_max_nli_scores[idx],
                "candidate_nli_scores": all_candidate_nli_scores[idx],
            }
            f.write(json.dumps(r) + "\n")
    with open(output_dir / "evaluation_result.json", 'w') as f:
        avg_context_f1 = calculate_average(all_context_f1)
        avg_context_precision = calculate_average(all_context_precision)
        avg_context_recall = calculate_average(all_context_recall)
        avg_max_nli_score = calculate_average(all_max_nli_scores)

        r = {
            "avg_context_f1": avg_context_f1,
            "avg_context_precision": avg_context_precision,
            "avg_context_recall": avg_context_recall,
            "avg_max_nli_score": avg_max_nli_score,
        }
        for metric in ["length", "rouge-1", "rouge-l", "bleu", "bleu-1", "bleu-2", "bleu-3", "bleu-4", "bertscore", "bartscore"]:
            if len(all_other_metrics[metric]) > 0:
                r[f"avg_{metric}"] = sum(all_other_metrics[metric]) / len(all_other_metrics[metric])
        f.write(json.dumps(r, indent=2) + "\n")

        



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_name')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_context_length', type=int, default=3600)
    parser.add_argument('--max_output_length', type=int, default=800)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--context_choice', default="Direct", choices=["DirectT", "Full", "CoTT", "RuleT1", "RuleT3", "RuleT5", "RuleT7", "RuleT9", "RuleT11", "RuleT13", "RuleV1", "RuleV3", "RuleV5", "RuleV7", "RuleMM5-3", "Oracle", "DirectMM", "FullVision", "DirectV", "CoTV", "CoTMM", "CombineMM"])
    parser.add_argument('--rewrite_choice', default="Yes", choices=["Yes", "No"])
    parser.add_argument('--verbose', type=int, default=2, help='Verbose level: 0 (quiet), 1 (error), 2 (info), 3 (debug)')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--api_key', type=str)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(args=parse_arguments())
    # demo()