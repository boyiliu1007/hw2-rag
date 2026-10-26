"""Reader: an open instruction-tuned LLM (served in-process by vLLM) that answers a question from
retrieved chunks, or closed-book with no context.

Answer length is a prompt choice (`style`): the metrics are token overlap (answer recall, F1,
ROUGE-1/2/L) plus an LLM judge. Few-shot questions come from the assignment handout's examples,
so no evaluation question appears in the prompt.
"""
import re

INTRO = ("You answer questions about Pittsburgh and Carnegie Mellon University (CMU). "
         "Today's date is {today}. Write dates like November 3, 2026, without the weekday. Copy names "
         "and numbers exactly as the passages write them; do not round amounts. ")
# Answer styles. Scoring is token overlap (recall, F1, ROUGE-1/2/L) plus an LLM judge, so the best
# answer contains the reference's words and little else.
STYLES = {
    "short": ("Reply with only the answer: a name, date, number, place or short phrase, usually "
              "under 10 words. Do not repeat the question, do not explain, and do not write a full "
              "sentence. For yes/no questions, start with Yes or No."),
    "phrase": ("Reply with a complete answer phrase: the answer together with the key words from the "
               "question it refers to, using the wording of the passages (for example the full name, "
               "the date with the event, or the amount with what it counts), usually 3 to 15 words. "
               "Do not add explanations or unrelated details. For yes/no questions, start with Yes or No."),
    "adaptive": ("If the question asks for a single fact (a name, number, year, date, time, place, "
                 "price, or yes/no), reply with just that fact and its essential qualifiers, using the "
                 "wording of the passages. If it asks you to describe or explain something (what is X, "
                 "what is the significance of, how did, why), reply with one concise sentence of at most "
                 "25 words that reuses the key terms of the question. Never add unrelated details. "
                 "For yes/no questions, start with Yes or No."),
}
# Definition/description questions get one sentence; everything else the short answer.
STYLES["sentence"] = ("Answer in one concise, complete sentence of at most 25 words that restates the "
                      "subject of the question and gives the key facts from the passages. Do not add "
                      "unrelated details.")
_VALUE = (r"name|date|year|address|phone|number|cost|price|fee|title|capacity|length|height|population|time|age|"
          r"percentage|rate|rank|ranking|amount|budget|revenue|score|record|distance|size|area|handle|username|"
          r"email|website|level|color|colors|role|position|mascot|nickname|motto|slogan|street|zip|location|"
          r"venue|day|days|month|season|team|opponent|artist|author|founder|owner|director|president|mayor|"
          r"coach|winner|genre|language|currency|abbreviation|acronym|county|city|state|neighborhood")
DESCRIPTIVE = re.compile(
    r"\bsignificance\b|\bimportance\b|"
    r"^how (did|does|has|have|was|were)\b(?!.*\b(acquir\w*|start|end|cost|long|many|much)\b)|"
    r"^why\b|\bwhat role\b|^describe\b|^explain\b|^what (makes|distinguishes)\b|"
    r"^what (is|was|are|were) (?:so )?(special|unique|notable|distinctive)\b|"
    rf"^what (is|was|are|were) (?!.*\b({_VALUE})\b)(?!.*\b(called|named|named after|located|held|built|founded)\s*\??$)"
    r"(?!.*\b(first|last|oldest|largest|longest|highest|tallest|main|only|most)\b)[^?]{3,80}\?\s*$", re.I)


# "full": like "routed", but factual answers keep their key qualifier instead of a bare value
STYLES["qualified"] = ("Reply with the answer together with its key qualifier from the passages, as a "
                       "phrase of about 2 to 8 words: the full name (not a surname), the time with its date, "
                       "the amount with what it counts, the acquisition type with its donor or fund. Use the "
                       "wording of the passages. Do not write a full sentence or add unrelated details. "
                       "For yes/no questions, start with Yes or No and add the reason in a few words.")
FEW_SHOT_QUALIFIED = [
    ("Who is Pittsburgh named after?", "William Pitt, 1st Earl of Chatham"),
    ("What famous machine learning venue had its first conference in Pittsburgh in 1980?",
     "ICML (International Conference on Machine Learning)"),
    ("What musical artist is performing at PPG Arena on October 13?", "Billie Eilish"),
]


# "restate": every answer is one short sentence that echoes the question's subject, the way
# sentence-style reference answers are written ("X performs in Pittsburgh on November 1, 2026.")
STYLES["restate"] = ("Answer with one short, complete sentence (at most 20 words) that restates the "
                     "subject of the question using the question's own words and then gives the answer "
                     "from the passages. Include nothing else: no extra dates, places or details that the "
                     "question did not ask for. For yes/no questions, start with Yes or No.")
FEW_SHOT_RESTATE = [  # the handout's example questions, answered in sentence form
    ("Who is Pittsburgh named after?", "Pittsburgh is named after William Pitt."),
    ("What famous machine learning venue had its first conference in Pittsburgh in 1980?",
     "ICML, the International Conference on Machine Learning, had its first conference in Pittsburgh in 1980."),
    ("What musical artist is performing at PPG Arena on October 13?",
     "Billie Eilish is performing at PPG Arena on October 13."),
]


def is_descriptive(question):
    """Definition/description questions (what is X, significance, how did, why), whose references
    tend to be full sentences."""
    return bool(DESCRIPTIVE.search(question.strip()))


RAG_RULES = (
    " Use the numbered context passages. If they disagree, prefer the passage that matches the "
    "question's names, dates and venues most exactly. Never say the answer is not specified or "
    "unknown: if the passages do not state it, give your single most likely answer."
)
FEW_SHOT = {  # handout example questions, or general ones; none of them in the evaluation sets
    "qualified": None,  # set below
    "sentence": [
        ("What is Picklesburgh?",
         "Picklesburgh is an annual pickle festival held in downtown Pittsburgh."),
        ("What is Light Up Night?",
         "Light Up Night is Pittsburgh's annual downtown celebration that kicks off the holiday season."),
        ("Why is Pittsburgh called the Steel City?",
         "Pittsburgh is called the Steel City because it was long the center of the American steel industry."),
    ],
    "short": [
        ("Who is Pittsburgh named after?", "William Pitt"),
        ("What famous machine learning venue had its first conference in Pittsburgh in 1980?", "ICML"),
        ("What musical artist is performing at PPG Arena on October 13?", "Billie Eilish"),
    ],
    "phrase": [
        ("Who is Pittsburgh named after?", "William Pitt, 1st Earl of Chatham"),
        ("What famous machine learning venue had its first conference in Pittsburgh in 1980?",
         "ICML, the International Conference on Machine Learning"),
        ("What musical artist is performing at PPG Arena on October 13?", "Billie Eilish"),
    ],
    "adaptive": [
        ("Who is Pittsburgh named after?", "William Pitt, 1st Earl of Chatham"),
        ("What musical artist is performing at PPG Arena on October 13?", "Billie Eilish"),
        ("Which three rivers meet in downtown Pittsburgh?",
         "The Allegheny, Monongahela and Ohio rivers"),
        ("What is Picklesburgh?",
         "Picklesburgh is an annual pickle festival held in downtown Pittsburgh."),
    ],
}


def format_context(chunks, max_words=1200):
    """Numbered passages, best first, cut to a total word budget."""
    out, used = [], 0
    for i, c in enumerate(chunks, 1):
        words = c["text"].split()
        if used + len(words) > max_words:
            words = words[:max(0, max_words - used)]
        if not words:
            break
        out.append(f"[{i}] {' '.join(words)}")
        used += len(words)
    return "\n\n".join(out)


def build_messages(question, chunks=None, today="October 24, 2026", max_words=None, style="short"):
    """Chat messages for one question; chunks=None means closed-book. style="routed" picks
    "sentence" for descriptive questions and "short" otherwise."""
    if style == "routed":
        style = "sentence" if is_descriptive(question) else "short"
    elif style == "full":
        style = "sentence" if is_descriptive(question) else "qualified"
    system = INTRO.format(today=today) + STYLES[style] + (RAG_RULES if chunks else "")
    msgs = [{"role": "system", "content": system}]
    for q, a in FEW_SHOT[style]:
        msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": a}]
    user = f"Question: {question}"
    if chunks:
        user = f"Context:\n{format_context(chunks, max_words or 240 * len(chunks))}\n\n{user}"
    msgs.append({"role": "user", "content": user})
    return msgs


def clean_answer(text):
    """Strip reasoning tags, labels and trailing punctuation the model sometimes adds."""
    if "</think>" in text:  # reasoning models: keep only what follows the reasoning
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.S).strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)  # citation markers like [2]
    return text.strip().strip('"“”').rstrip(".").strip()


REFUSAL = re.compile(
    r"\bnot (specified|mentioned|listed|provided|available|found|stated|given|known)\b|"
    r"^no (information|data|mention|answer|specific|such|record)\b|"
    r"^no \w+( \w+)? (is |are |was |were )?(listed|mentioned|found|scheduled|available|provided|given|specified|stated|known)\b|"
    r"\bunknown\b|cannot be determined|\bdoes not (say|specify|mention|state)\b|"
    r"\b(isn.t|is not|are not) (listed|specified|mentioned)\b|not in the (context|passages)", re.I)
GUESS = ("\n\nThe passages may not state this directly. You must still answer with your single most "
         "likely answer (a specific name, date, number or phrase). Never reply that it is not "
         "specified or unknown.")


def is_refusal(answer):
    return not answer.strip() or bool(REFUSAL.search(answer))


SHARED_HUB = "/data/hf_cache/hub"  # Babel's shared HF cache (read-only for us)


def resolve_model(model_id):
    """Use a complete, readable copy from the shared cluster cache if there is one; otherwise
    return the id so it is fetched into our own HF_HOME."""
    import glob
    import json
    import os
    for snap in sorted(glob.glob(f"{SHARED_HUB}/models--{model_id.replace('/', '--')}/snapshots/*")):
        idx = os.path.join(snap, "model.safetensors.index.json")
        try:
            files = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
        except (OSError, ValueError):
            continue
        if all(os.access(os.path.join(snap, f), os.R_OK) for f in files | {"config.json"}):
            return snap
    return model_id


class Reader:
    def __init__(self, model, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.88,
                 thinking=False, quantization=None):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=resolve_model(model), tensor_parallel_size=tensor_parallel_size,
                       quantization=quantization,
                       max_model_len=max_model_len if not thinking else max(max_model_len, 12288),
                       gpu_memory_utilization=gpu_memory_utilization, enforce_eager=False)
        if thinking:  # reasoning models: room to think; Qwen recommends light sampling over greedy
            self.params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=3000, seed=0)
        else:
            self.params = SamplingParams(temperature=0.0, max_tokens=64)

    def answer(self, message_lists, retry_refusals=True, fallback_lists=None):
        """Answer all questions. A refusal ("Not specified") always scores zero, so refusals are
        asked once more with an instruction to commit to a best guess, and if that still refuses,
        answered closed-book from `fallback_lists`."""
        outs = self.llm.chat(message_lists, self.params, use_tqdm=True)
        answers = [clean_answer(o.outputs[0].text) for o in outs]
        redo = [i for i, a in enumerate(answers) if is_refusal(a)]
        if retry_refusals and redo:
            retry = []
            for i in redo:
                msgs = [dict(m) for m in message_lists[i]]
                msgs[-1]["content"] += GUESS
                retry.append(msgs)
            for i, o in zip(redo, self.llm.chat(retry, self.params, use_tqdm=False)):
                guess = clean_answer(o.outputs[0].text)
                if guess and not is_refusal(guess):
                    answers[i] = guess
        still = [i for i, a in enumerate(answers) if is_refusal(a)]
        if fallback_lists and still:
            for i, o in zip(still, self.llm.chat([fallback_lists[i] for i in still], self.params, use_tqdm=False)):
                guess = clean_answer(o.outputs[0].text)
                if guess and not is_refusal(guess):
                    answers[i] = guess
        return answers


FEW_SHOT["qualified"] = FEW_SHOT_QUALIFIED
FEW_SHOT["restate"] = FEW_SHOT_RESTATE
