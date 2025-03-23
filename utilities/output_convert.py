import ast
import json

def model_output_to_json(model_output:str):
    json_obj = model_output.split("```")[1]
    if json_obj.startswith("json"):
        json_obj = json_obj[4:]
    json_obj = json_obj.strip()
    # print(json_obj)
    try:
        json_obj = json.loads(json_obj)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"retry using ast.literal_eval")
        try:
            json_obj = ast.literal_eval(json_obj)
            return json_obj
        except Exception as e:
            print(f"Error decoding JSON with ast.literal_eval: {e}")

