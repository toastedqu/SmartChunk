from transformers import T5Tokenizer, T5ForConditionalGeneration
import datasets
import evaluate
import typing
import functools
import chunker
import argparse
import retrieve


tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
exact_match_metric = evaluate.load("exact_match")

dataset: datasets.Dataset = datasets.load_dataset("squad", split="train")  # trivia_qa, natural_questions
dataset.shuffle()  # select 10000 samples
dataset = dataset.select(range(10000))


def run_model(input_string: str):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    seq = res["sequences"]  # identical to passes[i].argmax()
    passes = res["scores"]  # tuple(torch.FloatTensor), each element is a tensor of shape (1, vocab_size), by greedy search

    # get the probability of this generated sequence, by averaging the probability of each selected token
    # TODO: switch to max and other methods
    sum_prob = 0
    for p in passes:
        probs = p.softmax(dim=-1)
        selected_prob = probs.max()
        sum_prob += selected_prob
    avg_prob = sum_prob / len(passes)

    return tokenizer.batch_decode(seq, skip_special_tokens=True)[0], avg_prob


def run_model_top_l(question: str, chunks: typing.List[str], chunk_ids: typing.List[str], l: int, k: int = 20):
    for chunk, chunk_id in zip(chunks, chunk_ids):
        encodings = tokenizer.encode(question + " \\n " + chunk, return_tensors="pt")
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
    seq = res["sequences"]
    return tokenizer.batch_decode(seq, skip_special_tokens=True)[0]


def respond(question: str, context: str, chunker: typing.Callable, l: typing.Optional[int] = None):
    chunks, chunk_ids = chunker(doc=context, doc_id="context")  # chunk into splits
    answers = []
    probs = []
    if l is not None:
        answer = run_model_top_l(question, chunks, chunk_ids, l)
    else:
        for chunk in chunks:
            ans, prob = run_model(question + " \\n " + chunk)
            answers.append(ans)
            probs.append(prob)
        # choose the split answer with the highest probability
        max_prob = max(probs)
        max_prob_idx = probs.index(max_prob)
        answer = answers[max_prob_idx]
    return answer


def append_qa_answer(row, chunker_name: str = "whole_chunker", l: typing.Optional[int] = None):
    row["pred"] = respond(question=row["question"], context=row["context"], chunker=eval(chunker_name), l=l)
    row["ref"] = row["answers"]["text"][0]  # only take the first answer in the training set
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA-QA")
    parser.add_argument("-c", "--chunker", type=str, dest="chunker", default="chunker.whole_chunker", help="chunker function")
    parser.add_argument("-l", "--top-l", type=int, dest="topl", default=None, help="top l tokens to consider in EM")
    args = parser.parse_args()

    # DEBUG USE
    # args.chunker = "functools.partial(chunker.cluster_chunker, k=3, mode='k-preserve')"
    # args.topl = 100

    print("QA-QA")
    print("Arguments:", args)

    dataset = dataset.map(functools.partial(append_qa_answer, chunker_name=args.chunker, l=args.topl), batched=False)
    exact_match_result = exact_match_metric.compute(predictions=dataset["pred"], references=dataset["ref"])
    print("Results: ", exact_match_result)
