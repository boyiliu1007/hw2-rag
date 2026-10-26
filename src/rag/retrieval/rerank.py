"""Cross-encoder reranking: a first-stage retriever proposes `depth` candidates, then a
cross-encoder reads (question, passage) pairs jointly and reorders them.

Bi-encoders (dense) and BM25 score question and passage independently; a cross-encoder
attends across both, so it is better at deciding which of ~50 plausible passages actually
answers the question, at the cost of one model call per pair.

Two model families:
  bge     sequence-classification cross-encoders (sentence-transformers CrossEncoder)
  qwen3   Qwen3-Reranker: a causal LM asked whether the document answers the query; the score
          is P("yes") from the next-token logits, following the model card
"""

RERANKERS = {  # released before 2026-01-01
    "bge-reranker-v2-m3": ("bge", "BAAI/bge-reranker-v2-m3"),
    "bge-reranker-base": ("bge", "BAAI/bge-reranker-base"),
    "qwen3-reranker-0.6b": ("qwen3", "Qwen/Qwen3-Reranker-0.6B"),
    "qwen3-reranker-4b": ("qwen3", "Qwen/Qwen3-Reranker-4B"),
    "qwen3-reranker-8b": ("qwen3", "Qwen/Qwen3-Reranker-8B"),
}
QWEN3_TASK = "Given a question about Pittsburgh or Carnegie Mellon University, judge whether the passage answers it"


class _BgeScorer:
    def __init__(self, hf, device, max_length):
        import torch
        from sentence_transformers import CrossEncoder
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
        self.model = CrossEncoder(hf, device=device, max_length=max_length, **kw)

    def score(self, pairs, batch_size):
        return self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)


class _Qwen3Scorer:
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
              "the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
              "<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, hf, device, max_length):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tok = AutoTokenizer.from_pretrained(hf, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            hf, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, pairs, batch_size):
        out = []
        budget = self.max_length - len(self.pre) - len(self.suf)
        for i in range(0, len(pairs), batch_size):
            body = [f"<Instruct>: {QWEN3_TASK}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[i:i + batch_size]]
            enc = self.tok(body, add_special_tokens=False, truncation=True, max_length=budget)
            ids = [self.pre + x + self.suf for x in enc["input_ids"]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            two = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(self.torch.softmax(two, dim=1)[:, 1].tolist())
        return out


class RerankRetriever:
    def __init__(self, base, chunks, model="bge-reranker-v2-m3", depth=50, batch_size=None, max_length=512):
        self.base, self.chunks, self.depth = base, chunks, depth
        self.kind, self.hf = RERANKERS[model]
        self.batch_size = batch_size or (64 if self.kind == "bge" else 16)
        self.max_length = max_length
        self._scorer = None

    @property
    def scorer(self):
        if self._scorer is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls = _BgeScorer if self.kind == "bge" else _Qwen3Scorer
            self._scorer = cls(self.hf, device, self.max_length)
        return self._scorer

    def search(self, queries, k=10):
        cands = self.base.search(queries, k=self.depth)
        pairs, owners = [], []
        for qi, (q, hits) in enumerate(zip(queries, cands)):
            for idx, _ in hits:
                pairs.append((q, self.chunks[idx]["text"]))
                owners.append((qi, idx))
        scores = self.scorer.score(pairs, self.batch_size)
        out = [[] for _ in queries]
        for (qi, idx), s in zip(owners, scores):
            out[qi].append((idx, float(s)))
        return [sorted(o, key=lambda x: -x[1])[:k] for o in out]
