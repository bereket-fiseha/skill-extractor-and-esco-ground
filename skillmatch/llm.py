"""LLM refinement of raw NER spans (Qwen 2.5 Instruct, 4-bit quantised when possible)."""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ROOT
from .embedder import DEVICE

# A pre-downloaded copy in models/ is preferred (resumable download on flaky networks).
_LOCAL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
QWEN_MODEL = str(_LOCAL) if (_LOCAL / ".download_done").exists() else "Qwen/Qwen2.5-1.5B-Instruct"
ENTITIES_PER_CALL = 40
MAX_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = (
    "You are an expert HR data cleaner and entity resolution specialist. "
    "You will be provided with the original resume text and a raw list of extracted entities from a NER model. "
    "Your task is to refine this list by:\n"
    "1. Using the resume text as context to understand the entities.\n"
    "2. Removing entities that are clearly not skills, competencies, or assistive technologies.\n"
    "3. CRITICAL: Do NOT output fragmented, standalone verbs like 'developing', 'deployed', 'maintaining', "
    "or 'integrating'. You MUST convert them into canonical tech skills (e.g., 'Software Deployment', "
    "'Web Development') or remove them entirely if they lack context.\n"
    "4. Merging fragmented entities that should logically be a single skill (e.g., 'computer' and "
    "'engineering' -> 'computer engineering') based on the text.\n"
    "Output the result strictly as a comma-separated list of valid skills. "
    "Do not output any conversational text or explanations."
)


class QwenRefiner:
    def __init__(self, model_id: str = QWEN_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if DEVICE == "cuda":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # Turing GPUs (RTX 20xx) have no bf16
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    def _generate(self, resume_text: str, entities: list[str]) -> list[str]:
        user = (
            f"Resume Text:\n{resume_text[:MAX_CONTEXT_CHARS]}\n\n"
            f"Raw NER Entities: '{', '.join(entities)}'\n\nCleaned Output:"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=min(300, 10 * len(entities) + 30), do_sample=False)
        reply = self.tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return [s.strip(" \t\"'-•*.") for s in re.split(r"[,\n;]", reply) if s.strip(" \t\"'-•*.")]

    def refine(self, resume_text: str, entities: list[str]) -> list[str]:
        """Clean/merge NER entities. Falls back to the raw entities if the LLM returns nothing."""
        refined: dict[str, str] = {}
        for i in range(0, len(entities), ENTITIES_PER_CALL):
            for skill in self._generate(resume_text, entities[i : i + ENTITIES_PER_CALL]):
                refined.setdefault(skill.lower(), skill)
        return list(refined.values()) or list(entities)
