from collections import defaultdict
import pandas as pd
import logging
import json
import os

# ---------- M3EXAM Format ----------
# original from https://github.com/DAMO-NLP-SG/M3Exam
def generate_dev_examples(dev_questions, lang, method):
    dev_example_dict = defaultdict(lambda: defaultdict(list))
    for q in dev_questions:
        level = q['level']
        cate = q['subject_category']
        dev_string = generate_one_example(q, lang, method, fill_answer=True)
        dev_example_dict[level][cate].append(dev_string)
    
    return dev_example_dict

def generate_one_example(question, lang, method, fill_answer=False):
    answer_word = {
        'english': "Answer:", 'chinese': '答案：', 'vietnamese': 'Câu trả lời:', 'thai': 'คำตอบ:', 
        'italian': 'La risposta:', 'javanese': 'Wangsulan:', 'swahili': 'Jibu:', 
        'afrikaans': 'Antwoord:', 'portuguese': 'Responder:'
    }
    background = '\n' + '\n'.join(question['background_description']) if question['background_description'] else ''
    if method == 'default':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\n{answer_word[lang]}'
    elif method == 'en-instruct':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\nAnswer:'
    elif method == 'en-trans':
        prompt = question['background_description_english'] + '\n' + question['question_text_english'] + '\n' + question['options_english'] + f'\nAnswer:'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if fill_answer:
        prompt += str(question['answer_text'])
    
    return prompt

# added utility function for m3exam to get choices based on answer_text
def get_choices_m3exam(row : pd.Series):
    if row["answer_text"] in ['1','2','3','4','5']:
        return ['1','2','3','4','5'][:len(row["options"])]
    elif row["answer_text"] in ['๑', '๒', '๓', '๔', '๕']:
        return ['๑','๒','๓','๔','๕'][:len(row["options"])]
    else:
        logging.error(f"Unexpected answer_text: {row['answer_text']}. Please issue a support ticket.")
        return []
    
def generate_prompt(lang, method, setting, model, test_question, dev_question):
    subject2target = {
        'english': {'language': 'English', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'english4all': {'language': 'Language', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'chinese': {'language': '语文', 'math': "数学", 'social-science': "社会科学", 'natural-science': '自然科学'},
        'javanese': {'language': 'Bahasa Jawa'},
        'swahili': {'language': 'KISWAHILI'},
        'thai': {'language': 'ภาษาไทย', 'math': 'คณิตศาสตร์', 'social-science': 'สังคมศึกษา', 'natural-science': 'วิทยาศาสตร์'},
        'vietnamese': {'language': 'Tiếng Việt', 'math': "Toán", 'social-science': "Khoa học xã hội", 'natural-science': 'Khoa học tự nhiên'},
        'italian': {'language': 'Italiano', 'math': "Matematica", 'social-science': "Scienze sociali", 'natural-science': 'Scienze naturali'},
        'afrikaans': {'language': 'Afrikaans Huistaal', 'math': "Wiskunde", 'social-science': "Sosiale Wetenskappe", 'natural-science': 'Natuurwetenskap'},
        'portuguese': {'language': 'Linguagens', 'math': 'Matemática', 'social-science': 'Ciências Humanas', 'natural-science': 'Ciências da Natureza'},
    }
    subject = subject2target[lang][test_question['subject_category']]

    if method == 'default':
        hint = f"The following is a multiple choice question about {subject}."
        if model in ['chat', 'fake'] or setting == 'zero-shot':
            hint += ' Please only give the correct option, without any other details or explanations.'
    elif method in ['en-instruct', 'en-trans']:
        subject = subject2target['english4all'][test_question['subject_category']]
        hint = f"The following is a multiple choice question about {subject}."
        hint += ' Please only give the correct option, without any other details or explanations.'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if setting == 'zero-shot':
        prompt = hint + '\n\n' + generate_one_example(test_question, lang, method)
    elif setting == 'few-shot':
        dev_questions_list = dev_question[test_question['level']][test_question['subject_category']]
        prompt = hint + '\n\n' + '\n\n'.join(dev_questions_list) + '\n\n' + generate_one_example(test_question, lang, method)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    return prompt

DEV_EXAMPLES_CACHE = None
def load_dev_examples_once(lang="thai", method="default"):
    """Load dev examples into global cache if not already loaded."""
    global DEV_EXAMPLES_CACHE
    if DEV_EXAMPLES_CACHE is None:
        json_path = f"../data/m3exam/text-question/thai-questions-dev.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dev questions file not found: {json_path}")

        with open(json_path, "r") as f:
            dev_questions = json.load(f)

        DEV_EXAMPLES_CACHE = generate_dev_examples(dev_questions, lang, method)
        logging.info(f"Loaded dev examples: {sum(len(cats) for cats in DEV_EXAMPLES_CACHE.values())} categories")
    return DEV_EXAMPLES_CACHE