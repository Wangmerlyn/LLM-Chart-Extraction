import json

def model_output_to_json(model_output:str):
    json_obj = model_output.split("```")[1]
    if json_obj.startswith("json"):
        json_obj = json_obj[4:]
    json_obj = json_obj.strip()
    try:
        json_obj = json.loads(json_obj)
        return json_obj
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from model output.")
