import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration,AutoModelForImageTextToText

# 注意：qwen_vl_utils需要放在PYTHONPATH中或者与本文件放在同一目录下
from qwen_vl_utils import process_vision_info

"""
1、两个RL模型（Mulberry—llama和Mulberry-llava）在text_only的的情况下测试
2、Qwen2.5_VL_Instruct在text_only的情况下测试
"""


###############################################
# 模型封装接口定义
###############################################
# 1. Qwen2.5-VL-7B-Instruct
class Qwen2_5_VL_7B_Instruct:
    def __init__(self, model_id="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct/"):
        # 初始化模型和处理器
        self.model_id = model_id
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)


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
    
# 2. Mulberry_llama_11b
class Mulberry_llama_11b:
    def __init__(self, model_path = "/root/autodl-tmp/models/Mulberry_llama_11b"):
        self.model_id = model_path
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the "
            "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
            "tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
        )
    
    def infer(self, query):
        # 准备输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    # {"type": "image", "image": image_path},
                    {"type": "text", "text": f"{self.system_prompt}\n{query}"},
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

# 3. Mulberry_llava_8b
class Mulberry_llava_8b:
    def __init__(self, model_path = "/root/autodl-tmp/models/Mulberry_llava_8b"):
        self.model_id = model_path
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the "
            "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
            "tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
        )
    
    def infer(self, query):
        # image = Image.open(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{self.system_prompt}\n{query}"},
                # {"type": "image"},
            ]},
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

def run_experiments(
    questions_dir="../MM-SafetyBench/data/processed_questions",
    # questions_dir="/root/autodl-tmp/code/MM-SafetyBench/data/processed_questions",
   
    # images_dir="../MM-SafetyBench/data/imgs",
    output_dir="../MM-SafetyBench/questions_with_answers/baseline",
    # output_dir="/root/autodl-tmp/code/MM-SafetyBench/questions_with_answers/baseline",

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
        (Qwen2_5_VL_7B_Instruct, "Qwen2_5_VL_7B_Instruct"),
        (Mulberry_llama_11b, "Mulberry_llama_11b"),
        (Mulberry_llava_8b, "Mulberry_llava_8b"),
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


if __name__ == "__main__":
    run_experiments()
