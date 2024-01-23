from transformers import pipeline
import datasets
import evaluate

generator = pipeline("text2text-generation", model="allenai/unifiedqa-v2-t5-large-1251000")
dataset = datasets.load_dataset("squad", split="train[:20]")  # trivia_qa, natural_questions
exact_match_metric = evaluate.load("exact_match")

def get_answers(question: str, context: str):
    answers = generator(question + " \\n " + context)
    return answers

def append_qa_answer(row):
    answers = get_answers(row["question"], row["context"])
    row["pred"] = answers[0]["generated_text"]
    row["ref"] = row["answers"]["text"][0]  # only take the first answer in the training set
    return row

dataset = dataset.map(append_qa_answer, batched=False)

exact_match_result = exact_match_metric.compute(predictions=dataset["pred"], references=dataset["ref"])

print(exact_match_result)
