import json
import sys


def get_rough_crop(client, data_url, prompt_manager, model_name="gpt-4o") -> str:

    chat_completion = client.chat.completions.create(
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_manager.crop_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
        model=model_name,
    )

    return chat_completion.choices[0].message.content


def box_refine(client, data_url, prompt_manager, box_list) -> str:

    chat_completion = client.chat.completions.create(
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_manager.box_refine_prompt.format(
                            box_list
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
        model="gpt-4o",
    )
    message = chat_completion.choices[0].message.content
    return message


def extract_info(client, data_url, box_list, prompt_manager):
    """
    Extracts information from the sub-charts within an image.

    Parameters:
        client (OpenAI): An instance of the OpenAI class.
        data_url (str): The URL of the image data.
        prompt_manager (PromptManager): An instance of the PromptManager class.

    Returns:
        str: The extracted information.
    """

    chat_completion = client.chat.completions.create(
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_manager.extract_info_prompt.format(
                            box_list
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            }
        ],
        model="gpt-4o",
    )
    message = chat_completion.choices[0].message.content
    message
    return message


from agent_tools.get_top_colors import get_top_n_colors


def get_color(client, image_path, input_color, top_n=15) -> dict:
    get_top_n_colors(image_path, top_n)
    # Get the top N colors from the image
    top_colors = get_top_n_colors(image_path, top_n)

    # Print the top N most frequent colors and their counts
    color_prompt = f"""From the color list below, please identify the color that best matches the following description: {input_color}, return the RGB value in the format
    ```json{{
        "R": xxx,
        "G": xxx,
        "B": xxx,
    }}
    """
    for idx, (color, count) in enumerate(top_colors, 1):
        print(f"Color {idx}: R, G, B = {color}, Count = {count}")
        color_prompt += f"Color {idx}: R, G, B = {color}\n"
    color_message = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "user", "content": color_prompt},
        ],
    )
    color_message = color_message.choices[0].message.content
    print(color_message)
    color_message = color_message.split("```")[1]
    if color_message.startswith("json"):
        color_message = color_message[4:]
    color_message = color_message.strip()

    try:
        color_json = json.loads(color_message)
        return color_json
    except json.JSONDecodeError:
        print("Error decoding JSON: %s", color_message)
        sys.exit(1)
