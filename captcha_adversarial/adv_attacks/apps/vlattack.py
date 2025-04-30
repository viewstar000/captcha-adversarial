#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 viewstar000
"""
import sys
import json
import time
import threading
import gradio as gr


from ..targets.blip_attack import BlipBSAttacker, BlipVQAPredictorFactory, BlipBertAttacker
from ..targets.qwen_vl_attack import QwenVL25BSAttacker, QwenVL25Classifier, QwenVL25BertAttacker
from ..targets.openai_attack import OpenAIMultiModalClassifier


class StdoutCapture(object):

    def __init__(self, target, args=None, kwargs=None, max_buffer_size=4096):
        self.target = target
        self.args = args or tuple()
        self.kwargs = kwargs or dict()
        self.max_buffer_size = max_buffer_size
        self.started = False
        self.buffer = ""
        self.update_time = 0
        self.thread = None
        self.origin_stdout = sys.stdout

    def run(self):

        try:
            latest_update_time = self.update_time
            sys.stdout = self
            self.thread = threading.Thread(target=self.target, args=self.args, kwargs=self.kwargs, daemon=True)
            self.thread.start()
            self.started = True
            while True:
                time.sleep(1)
                if latest_update_time != self.update_time:
                    yield self.buffer[-1024:]
                    latest_update_time = self.update_time
                if not self.thread.is_alive():
                    break
        finally:
            sys.stdout = self.origin_stdout

    def write(self, text):
        self.origin_stdout.write(text)
        self.buffer = (self.buffer + text)[-self.max_buffer_size :]
        self.update_time = time.time()
        return len(text)

    def flush(self):
        pass


PREDICTORS = {
    "BlipVQAPredictorFactory": {
        "class": BlipVQAPredictorFactory,
        "configs": {"model_id": "Salesforce/blip-vqa-base"},
    },
    "QwenVL25Classifier": {
        "class": QwenVL25Classifier,
        "configs": {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    },
    "OpenAIMultiModalClassifier": {
        "class": OpenAIMultiModalClassifier,
        "configs": {
            "model_id": "qwen2.5-vl-7b-instruct",
            "api_base_url": "http://127.0.0.1:1234/v1",
            "api_key": "LMSTUDIO",
        },
    },
}

BSA_ATTACKERS = {
    "BlipBSAttacker": {"class": BlipBSAttacker},
    "QwenVL25BSAttacker": {"class": QwenVL25BSAttacker},
}

BERT_ATTACKERS = {
    "BlipBertAttacker": {"class": BlipBertAttacker},
    "QwenVL25BertAttacker": {"class": QwenVL25BertAttacker},
}


def switch_predictor_config(predictor_name):
    configs = PREDICTORS[predictor_name]["configs"]
    return (
        gr.Textbox(
            label="Model ID",
            value=configs.get("model_id"),
            visible="model_id" in configs,
            interactive=True,
        ),
        gr.Textbox(
            label="API Base URL",
            value=configs.get("api_base_url"),
            visible="api_base_url" in configs,
            interactive=True,
        ),
        gr.Textbox(
            label="API KEY",
            value=configs.get("api_key"),
            visible="api_key" in configs,
            interactive=True,
        ),
    )


def predictor_predict(predictor_name, model_id, api_base_url, api_key, image, text):

    def _predict():
        # Implement the prediction logic here
        print(f"Predicting with {predictor_name} ...")
        kwargs = {name: locals().get(name, default) for name, default in PREDICTORS[predictor_name]["configs"].items()}
        predictor_class = PREDICTORS[predictor_name]["class"]
        print(f"Create Predictor with {kwargs}")
        predictor = predictor_class(**kwargs)
        print("Predicting ...")
        label, probs = predictor.predict(text, image)
        print("=" * 20)
        print(f"Predicted: {label == 0} \n({probs[0].item()}, {probs[1].item()})")

    yield from StdoutCapture(_predict).run()


def bsa_attacker_execute(attacker_name, nb_iter, rand_init, image, text):

    result = {"image": None, "perturbation": None, "text": text}

    def _execute():

        print(f"Executing with {attacker_name} nb_iter: {nb_iter} rand_init: {rand_init}...")
        attacker = BSA_ATTACKERS[attacker_name]["class"](nb_iter=nb_iter, rand_init=rand_init)
        print("Attacking ...")
        outputs = attacker(image, text)
        result["image"] = outputs["adv_image"]
        result["perturbation"] = outputs["ptb_image"]
        print("Done.")

    for stdout in StdoutCapture(_execute).run():
        yield stdout, None, None, None
    yield stdout, result["image"], result["perturbation"], result["text"]


def bert_attacker_execute(attacker_name, similarity_threshold, image, text):

    result = {"candidates": [], "text": text}

    def _execute():

        print(f"Executing with {attacker_name} threshold: {similarity_threshold}...")
        attacker = BERT_ATTACKERS[attacker_name]["class"]()
        print("Attacking ...")
        outputs = attacker(text, image, similarity_threshold=similarity_threshold)
        result["candidates"] = json.dumps(outputs["candidates"], indent=2, ensure_ascii=False)
        result["text"] = outputs["text"]
        print("Done.")

    for stdout in StdoutCapture(_execute).run():
        yield stdout, None, None
    yield stdout, result["candidates"], result["text"]


def vla_attacker_execute(
    bsa_attacker_name,
    bert_attacker_name,
    origin_image,
    origin_text,
    pre_adv_image,
    bert_candidats,
    bsa_nb_iter,
    bsa_rand_init,
    vla_nb_iter,
):

    result = {"image": None, "perturbation": None, "text": None}

    def _execute():

        bsa_attacker = BSA_ATTACKERS[bsa_attacker_name]["class"](nb_iter=bsa_nb_iter, rand_init=bsa_rand_init)
        bert_attacker = BERT_ATTACKERS[bert_attacker_name]["class"]()
        origin_label, _ = bert_attacker.predictor_factory.predict(origin_text, origin_image)
        adv_image = pre_adv_image
        candidats = json.loads(bert_candidats)[:vla_nb_iter]
        for idx, (_, adv_text) in enumerate(candidats):
            print(f"[{idx+1:-3d}/{len(candidats)}] Adversarial text: {adv_text}")
            adv_outputs = bsa_attacker(adv_image, adv_text)
            result["image"] = adv_outputs["adv_image"]
            result["perturbation"] = adv_outputs["ptb_image"]
            result["text"] = adv_text
            adv_image = adv_outputs["adv_image"]
            label, probs = bert_attacker.predictor_factory.predict(adv_text, adv_image)
            print(f"[{idx+1:-3d}/{len(candidats)}] Adversarial label: {label.item()}, Adversarial probs: {probs}")
            if label != origin_label:
                print("Adversarial attack success!")
                return
        print("Adversarial attack failed!")

    for stdout in StdoutCapture(_execute).run():
        yield stdout, result["image"], result["perturbation"], result["text"]
    yield stdout, result["image"], result["perturbation"], result["text"]


with gr.Blocks(title="VLAttack Demo") as demo:
    gr.Markdown("## Welcome to VLAttack Demo")
    gr.Markdown("Stage 1: Select Taget Model")
    with gr.Row():
        with gr.Column():
            _ui_predictor_selector = gr.Dropdown(
                label="Target Predictor", choices=list(PREDICTORS.keys()), interactive=True, value=""
            )
            _ui_predictor_config_model_id = gr.Textbox(label="Model ID", visible=False)
            _ui_predictor_config_api_base_url = gr.Textbox(label="API Base URL", visible=False)
            _ui_predictor_config_api_key = gr.Textbox(label="API KEY", visible=False)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    _ui_origin_image = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    _ui_origin_prompt = gr.Textbox(label="Prompt Text")
                    _ui_origin_predict_result = gr.Textbox(
                        label="Predict Result", lines=3, max_lines=10, autoscroll=True
                    )
                    _ui_origin_predict_btn = gr.Button("Predict")
        _ui_predictor_selector.change(
            switch_predictor_config,
            inputs=[_ui_predictor_selector],
            outputs=[
                _ui_predictor_config_model_id,
                _ui_predictor_config_api_base_url,
                _ui_predictor_config_api_key,
            ],
        )
        _ui_origin_predict_btn.click(
            predictor_predict,
            inputs=[
                _ui_predictor_selector,
                _ui_predictor_config_model_id,
                _ui_predictor_config_api_base_url,
                _ui_predictor_config_api_key,
                _ui_origin_image,
                _ui_origin_prompt,
            ],
            outputs=_ui_origin_predict_result,
        )
    gr.Markdown("Stage 2: BSA Attack")
    with gr.Row():
        with gr.Column():
            _ui_bsa_attacker_selector = gr.Dropdown(
                label="BSA Attacker", choices=list(BSA_ATTACKERS.keys()), interactive=True
            )
            _ui_bsa_attacker_config_nb_iter = gr.Slider(
                label="Iteration Count", minimum=1, maximum=100, step=1, value=50, interactive=True
            )
            _ui_bsa_attacker_config_rand_init = gr.Checkbox(label="Random Init", value=True, interactive=True)
            _ui_bsa_attacker_exec_infos = gr.Textbox(
                label="Attack Infomations", lines=3, max_lines=10, autoscroll=True
            )
            _ui_bsa_attacker_exec_btn = gr.Button("Execute")
        with gr.Column():
            _ui_bsa_attacker_image_output = gr.Image(type="pil", label="Output Image", interactive=False)
        with gr.Column():
            _ui_bsa_attacker_perturbation_ouput = gr.Image(type="pil", label="Output Perturbation", interactive=False)
        with gr.Column():
            _ui_bsa_attacker_text_output = gr.Textbox(label="Prompt Text")
            _ui_bsa_attacker_predict_result = gr.Textbox(
                label="Predict Result", lines=3, max_lines=10, autoscroll=True
            )
            _ui_bsa_attacker_predict_btn = gr.Button("Predict")
        _ui_bsa_attacker_exec_btn.click(
            bsa_attacker_execute,
            inputs=[
                _ui_bsa_attacker_selector,
                _ui_bsa_attacker_config_nb_iter,
                _ui_bsa_attacker_config_rand_init,
                _ui_origin_image,
                _ui_origin_prompt,
            ],
            outputs=[
                _ui_bsa_attacker_exec_infos,
                _ui_bsa_attacker_image_output,
                _ui_bsa_attacker_perturbation_ouput,
                _ui_bsa_attacker_text_output,
            ],
        )
        _ui_bsa_attacker_predict_btn.click(
            predictor_predict,
            inputs=[
                _ui_predictor_selector,
                _ui_predictor_config_model_id,
                _ui_predictor_config_api_base_url,
                _ui_predictor_config_api_key,
                _ui_bsa_attacker_image_output,
                _ui_bsa_attacker_text_output,
            ],
            outputs=_ui_bsa_attacker_predict_result,
        )
    gr.Markdown("Stage 3: Bert Attack")
    with gr.Row():
        with gr.Column():
            _ui_bert_attacker_selector = gr.Dropdown(
                label="Bert Attacker", choices=list(BERT_ATTACKERS.keys()), interactive=True
            )
            _ui_bert_attacker_config_similarity_threshold = gr.Slider(
                label="Similarity Threshold", minimum=0, maximum=1, step=0.01, value=0.9, interactive=True
            )
            _ui_bert_attacker_exec_btn = gr.Button("Execute")
        with gr.Column():
            _ui_bert_attacker_exec_infos = gr.Textbox(
                label="Attack Infomations", lines=3, max_lines=10, autoscroll=True
            )
        with gr.Column():
            _ui_bert_attacker_candidates_output = gr.Textbox(
                label="Output Candidates", lines=3, max_lines=10, autoscroll=False
            )
        with gr.Column():
            _ui_bert_attacker_text_output = gr.Textbox(label="Prompt Text")
            _ui_bert_attacker_predict_result = gr.Textbox(
                label="Predict Result", lines=3, max_lines=10, autoscroll=True
            )
            _ui_bert_attacker_predict_btn = gr.Button("Predict")
        _ui_bert_attacker_exec_btn.click(
            bert_attacker_execute,
            inputs=[
                _ui_bert_attacker_selector,
                _ui_bert_attacker_config_similarity_threshold,
                _ui_origin_image,
                _ui_origin_prompt,
            ],
            outputs=[
                _ui_bert_attacker_exec_infos,
                _ui_bert_attacker_candidates_output,
                _ui_bert_attacker_text_output,
            ],
        )
        _ui_bert_attacker_predict_btn.click(
            predictor_predict,
            inputs=[
                _ui_predictor_selector,
                _ui_predictor_config_model_id,
                _ui_predictor_config_api_base_url,
                _ui_predictor_config_api_key,
                _ui_origin_image,
                _ui_bert_attacker_text_output,
            ],
            outputs=_ui_bert_attacker_predict_result,
        )
    gr.Markdown("Stage 4: VLA Union Attack")
    with gr.Row():
        with gr.Column():
            _ui_vla_attacker_config_vla_nb_iter = gr.Slider(
                label="VLAttack Iter Count", minimum=1, maximum=100, step=1, value=10, interactive=True
            )
            _ui_vla_attacker_config_bsa_nb_iter = gr.Slider(
                label="BSA Iter Count", minimum=1, maximum=100, step=1, value=5, interactive=True
            )
            _ui_vla_attacker_config_bsa_rand_init = gr.Checkbox(label="BSA Random Init", value=True, interactive=True)
            _ui_vla_attacker_exec_infos = gr.Textbox(
                label="Attack Infomations", lines=3, max_lines=10, autoscroll=True
            )
            _ui_vla_attacker_exec_btn = gr.Button("Execute")
        with gr.Column():
            _ui_vla_attacker_image_ouput = gr.Image(type="pil", label="Output Image", interactive=False)
        with gr.Column():
            _ui_vla_attacker_perturbation_ouput = gr.Image(type="pil", label="Output Perturbation", interactive=False)
        with gr.Column():
            _ui_vla_attacker_text_output = gr.Textbox(label="Prompt Text")
            _ui_vla_attacker_predict_result = gr.Textbox(
                label="Predict Result", lines=3, max_lines=10, autoscroll=True
            )
            _ui_vla_attacker_predict_btn = gr.Button("Predict")
        _ui_vla_attacker_exec_btn.click(
            vla_attacker_execute,
            inputs=[
                _ui_bsa_attacker_selector,
                _ui_bert_attacker_selector,
                _ui_origin_image,
                _ui_origin_prompt,
                _ui_bsa_attacker_image_output,
                _ui_bert_attacker_candidates_output,
                _ui_vla_attacker_config_bsa_nb_iter,
                _ui_vla_attacker_config_bsa_rand_init,
                _ui_vla_attacker_config_vla_nb_iter,
            ],
            outputs=[
                _ui_vla_attacker_exec_infos,
                _ui_vla_attacker_image_ouput,
                _ui_vla_attacker_perturbation_ouput,
                _ui_vla_attacker_text_output,
            ],
        )
        _ui_vla_attacker_predict_btn.click(
            predictor_predict,
            inputs=[
                _ui_predictor_selector,
                _ui_predictor_config_model_id,
                _ui_predictor_config_api_base_url,
                _ui_predictor_config_api_key,
                _ui_vla_attacker_image_ouput,
                _ui_vla_attacker_text_output,
            ],
            outputs=_ui_vla_attacker_predict_result,
        )

if __name__ == "__main__":
    demo.launch()
