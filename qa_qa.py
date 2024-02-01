from transformers import T5Tokenizer, T5ForConditionalGeneration
import datasets
import evaluate
import typing
import chunker


tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
model = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
dataset = datasets.load_dataset("squad", split="train[:20]")  # trivia_qa, natural_questions
exact_match_metric = evaluate.load("exact_match")


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

    return tokenizer.batch_decode(seq, skip_special_tokens=True), avg_prob


def respond(question: str, context: str, split: typing.Literal["none", "sent", "cluster", "arbitrary"] = "none", k: int = 4, l: typing.Optional[int] = None):
    if split == "none":
        chunks = [context]
    if split == "sent":
        # chunk into k-sentence splits
        chunks, _ = chunker.sent_cont_chunker(doc=context, doc_id=0, k=k)
    elif split == "cluster":
        # chunk into k cluster splits
        chunks, _ = chunker.cluster_chunker(doc=context, doc_id=0, k=k)
    elif split == "arbitrary":
        # chunk into k-word splits
        chunks, _ = chunker.arbitrary_chunker(doc=context, doc_id=0, k=k)
    else:
        raise ValueError("Invalid split method")
    answers = []
    probs = []
    for chunk in chunks:
        if l is not None and len(chunk.split(" ")) > l:
            chunk = " ".join(chunk.split(" ")[:l])  # truncate the chunk to length l tokens
        ans, prob = run_model(question + " \\n " + chunk)
        answers.append(ans[0])
        probs.append(prob)
    # choose the split answer with the highest probability
    max_prob = max(probs)
    max_prob_idx = probs.index(max_prob)
    answer = answers[max_prob_idx]
    return answer


def append_qa_answer(row):
    row["pred"] = respond(question=row["question"], context=row["context"], split="arbitrary", k=100)
    row["ref"] = row["answers"]["text"][0]  # only take the first answer in the training set
    return row


dataset = dataset.map(append_qa_answer, batched=False)

exact_match_result = exact_match_metric.compute(predictions=dataset["pred"], references=dataset["ref"])

print(exact_match_result)
