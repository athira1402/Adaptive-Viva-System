import itertools
import logging
import re
from typing import Optional, Dict, Union

from nltk import sent_tokenize
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

class QGPipeline:
    """Refined QG pipeline for Technical Contexts (Sorting/Searching)"""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer
        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def _clean_tech_text(self, text):
        """Standardizes algorithm notation for better model understanding"""
        text = re.sub(r'O\((.*?)\)', r'Big O of \1', text) 
        text = re.sub(r'(\w+)\[(\w+)\]', r'\1 element \2', text) 
        text = text.replace('log n', 'logarithmic n').replace('n^2', 'n squared')
        return " ".join(text.split())

    def process_viva_text(self, input_text, chunk_size=3):
        """
        Processes long text by chunking it semantically to maintain context 
        and feeding it into the pipeline.
        """
        sentences = sent_tokenize(input_text)
        chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        
        all_results = []
        for chunk in chunks:
            try:
                results = self.__call__(chunk)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
        return all_results

    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, answers = self._extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))
        
        if len(flat_answers) == 0:
          return []

        if self.qg_format == "prepend":
            qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
        else:
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)
        
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=64,
            num_beams=8, # High beam search for viva accuracy
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)
        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
        )
        
        # skip_special_tokens=True fixed the <pad> issue
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        answers = [item.split('<sep>') for item in dec]
        answers = [i[:-1] if i[-1] == '' else i for i in answers]
        
        return sents, answers
    
    def _tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=512):
        inputs = self.tokenizer(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)
        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, self._clean_tech_text(sent))
            
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text.strip())
        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sents_copy = [self._clean_tech_text(s) for s in sents]
                clean_ans = self._clean_tech_text(answer_text.strip())
                clean_sent = sents_copy[i]
                
                # Highlight logic with fallback to ensure <hl> tags are present
                try:
                    if clean_ans in clean_sent:
                        clean_sent = clean_sent.replace(clean_ans, f"<hl> {clean_ans} <hl>", 1)
                        sents_copy[i] = clean_sent
                except Exception:
                    pass 
                
                source_text = "generate question: " + " ".join(sents_copy)
                if self.model_type == "t5":
                    source_text += " </s>"
                inputs.append({"answer": answer_text, "source_text": source_text})
        return inputs

    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        clean_context = self._clean_tech_text(context)
        for answer in flat_answers:
            clean_ans = self._clean_tech_text(answer)
            source_text = f"answer: {clean_ans} context: {clean_context}"
            if self.model_type == "t5":
                source_text += " </s>"
            examples.append({"answer": answer, "source_text": source_text})
        return examples

class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer(
    	inputs,
    	max_length=max_length,
    	add_special_tokens=add_special_tokens,
    	truncation=truncation,
    	padding="max_length" if padding else True,
    	return_tensors="pt"
)

        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "valhalla/t5-small-qg-hl",
            "ans_model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "multitask-qa-qg": {
        "impl": MultiTaskQAQGPipeline,
        "default": {
            "model": "valhalla/t5-small-qa-qg-hl",
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-small-e2e-qg",
        }
    }
}

def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
            ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

            if isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    elif task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, use_cuda=use_cuda)
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer, qg_format=qg_format, use_cuda=use_cuda)
