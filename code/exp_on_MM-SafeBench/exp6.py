import os
import json
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# 注意：qwen_vl_utils需要放在PYTHONPATH中或者与本文件放在同一目录下
from qwen_vl_utils import process_vision_info

"""
1、Llama-3.2-11B-Vision-Instruct在text_only的的情况下测试
2、Llama-3.2-11B-Vision-Instruct在jailbreak的的情况下测试
"""


###############################################
# 模型封装接口定义
###############################################
# 1. Llama-3.2-11B-Vision-Instruct
class Llama_3_2_11B_Vision_Instruct_text_only:
    def __init__(self, model_path="/root/autodl-tmp/models/Llama-3.2-11B-Vision-Instruct"):
        self.model_id = model_path
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer(self, query):
        # 准备输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        # 处理消息
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 执行推理生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 返回生成的文本
        return output_text[0]

# 2. 
class Llama_3_2_11B_Vision_Instruct_jailbreak:
    def __init__(self, model_path="/root/autodl-tmp/models/Llama-3.2-11B-Vision-Instruct"):
        self.model_id = model_path
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer(self, image_path, query):
        # 准备输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": query},
                ],
            }
        ]
        # 处理消息
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 执行推理生成输出
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 返回生成的文本
        return output_text[0]


###############################################
# 实验代码：遍历 MM-SafetyBench 某个场景，对四个模型进行推理
###############################################

def run_experiments_text_only(
    questions_dir="../MM-SafetyBench/data/processed_questions",
    # images_dir="../MM-SafetyBench/data/imgs",
    output_dir="../MM-SafetyBench/questions_with_answers/baseline",
    # image_type="SD_TYPO"  # 可选 "SD", "SD_TYPO", "TYPO"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有场景的问答 JSON 文件（例如 "01-Illegal_Activitiy.json"）
    # 我们测试：[1,2,3,4,6,7,9,10,11,12]
    scenario_files = sorted(os.listdir(questions_dir))
    # 从scenario_files取出第1,2,3,4,6,7,9,10,11,12个文件
    scenario_files = [scenario_files[i] for i in [0,1,2,3,5,6,8,9,10,11]]
    
    # 逐个加载模型并处理所有场景
    models = [
        (Llama_3_2_11B_Vision_Instruct_text_only, "Llama_3_2_11B_Vision_Instruct_text_only"),
    ]
    
    # 对每个模型分别进行推理
    for model_class, model_name in models:
        print(f"Processing with model: {model_name}")
        if not os.path.exists(os.path.join(output_dir, model_name)):
            os.makedirs(os.path.join(output_dir, model_name))
            
        # 实例化模型
        model = model_class()

        for scenario_file in scenario_files:
            scenario_name = os.path.splitext(scenario_file)[0]
            scenario_path = os.path.join(questions_dir, scenario_file)
            print(f"Processing scenario: {scenario_name}")
            
            with open(scenario_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            
            scenario_results = {}
            
            for qid, qdata in questions.items():
                query = qdata.get("Changed Question", "")
                print(f"  Using query: {query}")
                scenario_results[qid] = {
                    "Question": qdata.get("Question", ""),
                    "Changed Question": qdata.get("Changed Question", ""),
                    "Rephrased Question": query,
                    "answers": {}
                }
                
                print(f"  Processing question id {qid} ...")
                try:
                    ans = model.infer(query)
                except Exception as e:
                    ans = f"Error: {e}"
                
                scenario_results[qid]["answers"][model_name] = ans
                
                if ans:
                    print(f"question: 【{qid}】 finished!")
                else:
                    print(f"question: 【{qid}】 failed!")
                    continue
                
                # break # 仅处理第一个问题以节省时间

            # 保存当前场景的答案到 output 目录中
            
            output_path = os.path.join(output_dir, model_name, scenario_file)
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(scenario_results, out_f, ensure_ascii=False, indent=4)
            print(f"Saved results for {scenario_name} to {output_path}\n")
            # break # 仅处理第一个场景以节省时间
        # 释放显存
        del model
        torch.cuda.empty_cache()
        # break # 仅处理第一个模型以节省时间

def run_experiments_jailbreak(
    questions_dir="../MM-SafetyBench/data/processed_questions",
    images_dir="../MM-SafetyBench/data/imgs",
    output_dir="../MM-SafetyBench/questions_with_answers/jailbreak_method",
    image_type="SD_TYPO"  # 可选 "SD", "SD_TYPO", "TYPO"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有场景的问答 JSON 文件（例如 "01-Illegal_Activitiy.json"）
    # 我们测试：[1,2,3,4,6,7,9,10,11,12]
    scenario_files = sorted(os.listdir(questions_dir))
    # 从scenario_files取出第1,2,3,4,6,7,9,10,11,12个文件
    scenario_files = [scenario_files[i] for i in [0,1,2,3,5,6,8,9,10,11]]
    
    # 逐个加载模型并处理所有场景
    models = [
        (Llama_3_2_11B_Vision_Instruct_jailbreak, "Llama_3_2_11B_Vision_Instruct_jailbreak"),
        (Llama_3_2_11B_Vision_Instruct_jailbreak, "Llama_3_2_11B_Vision_Instruct_jailbreak"),
    ]
    
    # 对每个模型分别进行推理
    for model_class, model_name in models:
        print(f"Processing with model: {model_name}")
        if not os.path.exists(os.path.join(output_dir, model_name)):
            os.makedirs(os.path.join(output_dir, model_name))
            
        # 实例化模型
        model = model_class()

        for scenario_file in scenario_files:
            scenario_name = os.path.splitext(scenario_file)[0]
            scenario_path = os.path.join(questions_dir, scenario_file)
            print(f"Processing scenario: {scenario_name}")
            
            with open(scenario_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            
            scenario_results = {}
            
            for qid, qdata in questions.items():
            # for qid, qdata in list(questions.items())[:1]:  # 只取前一条数据
                # 构造图像路径：例如 ../MM-SafetyBench/data/imgs/{scenario}/{image_type}/{qid}.jpg
                image_path = os.path.join(images_dir, scenario_name, image_type, f"{qid}.jpg")
                print(f"  Image path: {image_path}")
                # 选择使用的问句，这里采用 "Rephrased Question" 字段
                query = qdata.get("Rephrased Question", "")
                print(f"  Using query: {query}")
                scenario_results[qid] = {
                    "Question": qdata.get("Question", ""),
                    "Changed Question": qdata.get("Changed Question", ""),
                    "Rephrased Question": query,
                    "answers": {}
                }
                
                print(f"  Processing question id {qid} ...")
                try:
                    ans = model.infer(image_path, query)
                except Exception as e:
                    ans = f"Error: {e}"
                
                scenario_results[qid]["answers"][model_name] = ans
                
                if ans:
                    print(f"question: 【{qid}】 finished!")
                else:
                    print(f"question: 【{qid}】 failed!")
                    continue
                
                # break # 仅处理第一个问题以节省时间

            # 保存当前场景的答案到 output 目录中
            
            output_path = os.path.join(output_dir, model_name, scenario_file)
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(scenario_results, out_f, ensure_ascii=False, indent=4)
            print(f"Saved results for {scenario_name} to {output_path}\n")
            # break # 仅处理第一个场景以节省时间
        # 释放显存
        del model
        torch.cuda.empty_cache()
        # break # 仅处理第一个模型以节省时间

if __name__ == "__main__":
    run_experiments_text_only()
    run_experiments_jailbreak()

