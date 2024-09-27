import re
import json
import argparse
import concurrent
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)
from statistics import multimode

import textgrad as tg
from textgrad.tasks import load_instance_task


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="MMLU_machine_learning", help="The task to evaluate the model on.")
    parser.add_argument("--engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--max_iterations", type=int, default=3, help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--num_threads", type=int, default=16, help="The number of threads to use for evaluation.")
    return parser.parse_args()


class MajorityVoting:
    def __init__(self):
        pass

    def __call__(self, predictions):
        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred_labels = []
        for pred in predictions:
            match = re.search(ANSWER_PATTERN_MULTICHOICE, pred.value)
            extracted_answer = match.group(1) if match else None
            pred_labels.append(extracted_answer)
        
        modes = multimode(pred_labels)
        return tg.Variable(f"Answer: {modes[0]}", role_description="Majority ensemble")


def get_zeroshot_answer(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))
    return response

def get_zeroshot_answer2(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    # The system prompt is from: https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    STARTING_SYSTEM_PROMPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-09-26"
)
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False, role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))
    return response

def extract_answer(paragraph):
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, paragraph, re.DOTALL)
    return matches

def run_test_time_training(sample):
    GRADIENT_TEMPLATE = """
    Here is answer1: {answer1}
    Here is evaluation1: {evaluation1}
    Here is answer2: {answer2}
    Here is evaluation2: {evaluation2}
    """

    OPTIMIZATION_TEMPLATE = """
    Here is answer1: {answer1}
    Here is evaluation1: {evaluation1}
    Here is answer2: {answer2}
    Here is evaluation2: {evaluation2}
    Here is the analysis: {gradient}
    """

    GRADIENT_COMPUTOR_SYSTEM_PROMPT = """
    You are part of an optimization system that improves a given answer (i.e. answer2). You are the gradient (feedback) engine.
    Your only responsibility is to give intelligent and creative feedback and constructive criticism to variables.
    You will be given two answers to a same question (answer1 and answer2), and the evaluation for each answer (evaluation1 and evaluation2).
    Analyze how the difference in the two evaluations(evaluation2 - evaluation1) are caused by the difference in the answers (answer2 - answer1).
    """
    gradient_computor = tg.BlackboxLLM(llm_engine, GRADIENT_COMPUTOR_SYSTEM_PROMPT)

    OPTIMIZER_SYSTEM_PROMPT = """
    You are part of an optimization system that improves text (i.e., answer2).
    You will receive two answers to a question (answer1 and answer2), two evaluations for each answer (evaluation1 and evaluation2) and an analysis of how the evaluation over answer2 being different from answer1 (evaluation2 - evaluation1) is caused by the difference in the answers (answer2 - answer1).
    Based on the analysis, you should provide a new version of answer2 that improves the worse aspects of answer2 and keep the better aspects of answer2 comparing to answer1.
    You can do this step by step, but make sure finally you provide the new version of answer2 between <answer> and </answer> tags.
    """
    optimizer = tg.BlackboxLLM(llm_engine, OPTIMIZER_SYSTEM_PROMPT)

    performance_history = []
    question, answer, test_time_objective, instance_eval_fn = sample
    zero_shot_response1 = get_zeroshot_answer(question)
    
    instance_var1 = tg.Variable(zero_shot_response1.value,
                               requires_grad=False,
                               role_description="creative and precise solution and the prediction for the multiple choice question")
    
    # Evaluate the zero-shot response
    evaluation_1 = test_time_objective(instance_var1)


    zero_shot_response2 = get_zeroshot_answer2(question)
    instance_var2 = tg.Variable(zero_shot_response2.value,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")
    performance_history.append(int(instance_eval_fn(instance_var2)))
    predictions = []
    predictions.append(tg.Variable(
        instance_var2.value,
        role_description=instance_var2.role_description
        ))

    evaluation_2 = test_time_objective(instance_var2)

    gradient = gradient_computor(tg.Variable(GRADIENT_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value, answer2=instance_var2.value, evaluation2=evaluation_2.value), role_description="gradient computation"))

    print('*' * 50)
    print('Optimization round 1')
    print(f'Gradient: {gradient.value}')

    optimizer_response = optimizer(tg.Variable(OPTIMIZATION_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value, answer2=instance_var2.value, evaluation2=evaluation_2.value, gradient=gradient.value), role_description="optimizer response"))

    print(f'Optimizer response: {optimizer_response.value}')
    print('*' * 50)


    new_answer = extract_answer(optimizer_response.value)

    if len(new_answer) == 0:
        new_answer = optimizer_response.value
        # print(f'No new answer: {new_answer}')
    else:
        new_answer = new_answer[0]

    # iteration step
    instance_var1 = tg.Variable(instance_var2.value,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")
    instance_var2 = tg.Variable(new_answer,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")
    performance_history.append(int(instance_eval_fn(instance_var2)))

    evaluation_1 = test_time_objective(instance_var1)

    predictions = []
    predictions.append(tg.Variable(
        instance_var2.value,
        role_description=instance_var2.role_description
    ))

    evaluation_2 = test_time_objective(instance_var2)

    gradient = gradient_computor(tg.Variable(
        GRADIENT_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value,
                                 answer2=instance_var2.value, evaluation2=evaluation_2.value),
        role_description="gradient computation"))

    print('*' * 50)
    print('Optimization round 2')
    print(f'Gradient: {gradient.value}')

    optimizer_response = optimizer(tg.Variable(
        OPTIMIZATION_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value,
                                     answer2=instance_var2.value, evaluation2=evaluation_2.value,
                                     gradient=gradient.value), role_description="optimizer response"))


    print(f'Optimizer response: {optimizer_response.value}')
    print('*' * 50)

    new_answer = extract_answer(optimizer_response.value)

    if len(new_answer) == 0:
        new_answer = optimizer_response.value
        # print(f'No new answer: {new_answer}')
    else:
        new_answer = new_answer[0]


    # iteration step
    instance_var1 = tg.Variable(instance_var2.value,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")
    instance_var2 = tg.Variable(new_answer,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")
    performance_history.append(int(instance_eval_fn(instance_var2)))

    evaluation_1 = test_time_objective(instance_var1)

    predictions = []
    predictions.append(tg.Variable(
        instance_var2.value,
        role_description=instance_var2.role_description
    ))

    evaluation_2 = test_time_objective(instance_var2)

    gradient = gradient_computor(tg.Variable(
        GRADIENT_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value,
                                 answer2=instance_var2.value, evaluation2=evaluation_2.value),
        role_description="gradient computation"))

    print('*' * 50)
    print('Optimization round 3')
    print(f'Gradient: {gradient.value}')

    optimizer_response = optimizer(tg.Variable(
        OPTIMIZATION_TEMPLATE.format(answer1=instance_var1.value, evaluation1=evaluation_1.value,
                                     answer2=instance_var2.value, evaluation2=evaluation_2.value,
                                     gradient=gradient.value), role_description="optimizer response"))


    print(f'Optimizer response: {optimizer_response.value}')
    print('*' * 50)

    new_answer = extract_answer(optimizer_response.value)

    if len(new_answer) == 0:
        new_answer = optimizer_response.value
        # print(f'No new answer: {new_answer}')
    else:
        new_answer = new_answer[0]

    instance_var2 = tg.Variable(new_answer,
                                requires_grad=False,
                                role_description="creative and precise solution and the prediction for the multiple choice question")

    performance_history.append(int(instance_eval_fn(instance_var2)))
    predictions.append(tg.Variable(
        instance_var2.value,
        role_description=instance_var2.role_description
    ))

    ensembled_prediction = ensembler(predictions)
    performance_history.append(instance_eval_fn(ensembled_prediction))
    predictions.append(ensembled_prediction)

    return performance_history, predictions, question, answer



args = config()
args.task = 'GPQA_main'
args.engine = 'azure-gpt4o'
args.num_threads = 1
llm_engine = tg.get_engine(engine_name=args.engine)
tg.set_backward_engine(llm_engine, override=True)
test_set = load_instance_task(args.task, evaluation_api=llm_engine, max_samples=10)
ensembler = MajorityVoting()

all_solutions = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    futures = []
    for _, sample in enumerate(test_set):
        future = executor.submit(run_test_time_training, sample)
        futures.append(future)

    all_history = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
        performance_history, predictions, question, answer = future.result()
        # assert len(performance_history) == args.max_iterations + 2
        # assert len(predictions) == args.max_iterations + 2
        all_solutions[question] = {"predictions": [p.value for p in predictions], "answer": answer}
        all_history.append(performance_history)

print(np.array(all_history).mean(axis=0))
with open(f"./{args.task}_predictions_new.json", "w") as f:
    json.dump(all_solutions, f)
