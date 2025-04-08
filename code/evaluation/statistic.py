import os
import json
from collections import defaultdict


def main():
    # 评估结果的路径
    eval_results_saved_path = "./evaluation_results/MM-SafetyBench/"
    # methods
    methods = ["baseline", "jailbreak_method"]
    
    # 统计结果保存的路径    
    statistic_results_saved_path = "./statistic_results"

    # 所有可能的分数值
    possible_scores = [5,4,3,2,1,0,-1]

    for method in methods:
        # 模型列表
        models = sorted(os.listdir(os.path.join(eval_results_saved_path, method)))
        # print(models)
        for model in models:
            scenario_files = sorted(os.listdir(os.path.join(eval_results_saved_path, method, model)))
            # print(scenario_files)
            # break
            for scenario_file in scenario_files:
                # 标记一下现在在处理哪个scenario
                scenario_name = os.path.splitext(scenario_file)[0]
                print(f"Processing scenario: {scenario_name}")
                
                with open(os.path.join(eval_results_saved_path, method, model, scenario_file), 'r', encoding="utf-8") as f:
                    eval_results = json.load(f)

                # 初始化存储结果的字典，确保每个分数值从-1到5都初始化为0
                result = {
                    "reasoning_score": {score: 0 for score in possible_scores},
                    "answer_score": {score: 0 for score in possible_scores},
                    "full_response_score": {score: 0 for score in possible_scores}
                }

                # 遍历数据集，统计每个score的数量
                for key, entry in eval_results.items():  # 遍历字典
                    # 确保每个条目是字典，且包含正确的键
                    if isinstance(entry, dict) and all(k in entry for k in ["reasoning_score", "answer_score", "full_response_score"]):
                        reasoning_score = entry["reasoning_score"]
                        answer_score = entry["answer_score"]
                        full_response_score = entry["full_response_score"]
                        
                        # 增加对应分数的计数
                        result["reasoning_score"][reasoning_score] += 1
                        result["answer_score"][answer_score] += 1
                        result["full_response_score"][full_response_score] += 1
                    else:
                        print(f"Invalid entry format: {entry}")

                # 保存到 result.json 文件
                output_dir = os.path.join(statistic_results_saved_path, method, model)
                os.makedirs(output_dir, exist_ok=True)  # 确保保存路径存在
                with open(os.path.join(output_dir, scenario_file), "w", encoding="utf-8") as outfile:
                    json.dump(result, outfile, indent=4, ensure_ascii=False)

                print(f"统计结果已保存到 {os.path.join(output_dir, scenario_file)}")
        #         break
        #     break
        # break


if __name__ == "__main__":
    main()
