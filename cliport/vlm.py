import logging
from pathlib import Path
from typing import Union, List

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms


def load_vlm(vlm_name: str, device: Union[torch.device, str]):
    # loads InstructBLIP pre-trained model
    logging.info("-" * 10 + f"Loading the VLM {vlm_name}" + "-" * 10)
    if vlm_name == "OpenFlamingo-3B-vitl-mpt1b":
        vlm, vis_processors, txt_processors = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1
        )
        vlm.to(device)
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b",
                                          "checkpoint.pt")
        vlm.load_state_dict(torch.load(checkpoint_path), strict=False)
    elif vlm_name == "OpenFlamingo-9B-vitl-mpt7b":
        vlm, vis_processors, txt_processors = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4
        )
        vlm.to(device)
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b",
                                          "checkpoint.pt")
        vlm.load_state_dict(torch.load(checkpoint_path), strict=False)
    else:
        raise ValueError("vlm_name is set right!")

    return vlm, vis_processors, txt_processors


def get_prompt_for_openflamingo(
        tokenizer,
        img_processor,
        example_image_directory: str,
        test_images: List[str],
        is_success_description: bool = False,
        last_instruction: str = None,
        device: Union[torch.device, str] = None
):
    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left

    # For OpenFlamingo, we expect the image to be a torch tensor of shape
    #  batch_size x num_media x num_frames x channels x height x width.
    vision_x = []
    img_list = [x for x in Path(example_image_directory).iterdir() if x.suffix in [".png", ".PNG"]]
    for i in range(1, len(img_list) + 1):
        img_path = Path(example_image_directory) / f"{i}.png"
        image = Image.open(img_path).convert("RGB")
        vision_x.append(img_processor(image).unsqueeze(0))

    if not is_success_description:
        assert len(test_images) == 1
        image = Image.open(test_images[0]).convert("RGB")
        vision_x.append(img_processor(image).unsqueeze(0))
    else:
        assert len(test_images) == 2
        for image in test_images:
            image = Image.open(image).convert("RGB")
            vision_x.append(img_processor(image).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    if not is_success_description:
        # lang_x = tokenizer(
        #     [
        #         "Please describe the position of each block on the table. Please clarify the color of each object."
        #         "<image>On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. Other bowls are empty. <|endofchunk|>"
        #         "<image>On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. The blue block is in the orange bowl. "
        #         "Other bowls are empty. <|endofchunk|>"
        #         "<image>On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table. <|endofchunk|>"
        #         "<image>On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table. "
        #         "The green block is in the purple zone. <|endofchunk|>"
        #         "<image>On the current scene,"
        #     ],
        #     return_tensors="pt",
        # )
        # lang_x = tokenizer(
        #     [
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. The blue block is in the orange bowl. "
        #         "Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. The blue block is in the orange bowl. "
        #         "Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table. "
        #         "The green block is in the purple zone.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. The blue block is in the orange bowl. "
        #         "Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
        #         "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
        #         "The white block is in the white bowl. The blue block is in the orange bowl. "
        #         "Other bowls are empty.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #         "On the current scene, there are a red block and a green block on the table. "
        #         "There are a white bowl and a blue bowl on the table. All the bowls are empty."
        #         "There are a purple zone, a white zone, and a green zone on the table. "
        #         "The green block is in the purple zone.<|endofchunk|>"
        #         "<image>Question: What objects are on the table and what's the position of each block? Answer: "
        #     ],
        #     return_tensors="pt",
        # )

        lang_x = tokenizer(
            [
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. The blue block is in the orange bowl. "
                "Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. The blue block is in the orange bowl. "
                "Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are a red block and a green block on the table. "
                "There are a white bowl and a blue bowl on the table. All the bowls are empty."
                "There are a purple zone, a white zone, and a green zone on the table.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are a red block and a green block on the table. "
                "There are a white bowl and a blue bowl on the table. All the bowls are empty."
                "There are a purple zone, a white zone, and a green zone on the table. "
                "The green block is in the purple zone.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. The blue block is in the orange bowl. "
                "Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table. "
                "There are white, gray, purple, brown, orange, blue, and pink bowls on the table. "
                "The white block is in the white bowl. The blue block is in the orange bowl. "
                "Other bowls are empty.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are a red block and a green block on the table. "
                "There are a white bowl and a blue bowl on the table. All the bowls are empty."
                "There are a purple zone, a white zone, and a green zone on the table.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
                "On the current scene, there are a red block and a green block on the table. "
                "There are a white bowl and a blue bowl on the table. All the bowls are empty."
                "There are a purple zone, a white zone, and a green zone on the table. "
                "The green block is in the purple zone.<|endofchunk|>"
                "<image>What objects are on the table and what's the position of each block? Answer: "
            ],
            return_tensors="pt",
        )
        assert vision_x.size(1) == 5
    else:
        assert last_instruction is not None
        lang_x = tokenizer(
            [
                # "Please compare the two images and judge whether the robot's action is executed successfully."
                "<image><image>"
                "Question: Is the last step instruction 'pick up the blue block and place it in the orange bowl' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the blue block and place it in the orange bowl' "
                "is executed successfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the blue block and place it in the blue bowl' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the blue block and place it in the blue bowl' "
                "is executed unsuccessfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the green block and place it in the purple zone' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the green block and place it in the purple zone' "
                "is executed successfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the green block and place it in the green zone' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the green block and place it in the green zone' "
                "is executed unsuccessfully. "
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the blue block and place it in the orange bowl' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the blue block and place it in the orange bowl' "
                "is executed successfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the blue block and place it in the blue bowl' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the blue block and place it in the blue bowl' "
                "is executed unsuccessfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the green block and place it in the purple zone' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the green block and place it in the purple zone' "
                "is executed successfully."
                "<|endofchunk|>"
                "<image><image>"
                "Question: Is the last step instruction 'pick up the green block and place it in the green zone' "
                "executed successfully? "
                "Answer: The last step instruction 'pick up the green block and place it in the green zone' "
                "is executed unsuccessfully. "
                "<|endofchunk|>"
                "<image><image>"
                f"Question: Is the last step instruction '{last_instruction}' executed successfully? "
                "Answer: "
            ],
            return_tensors="pt",
        )
        assert vision_x.size(1) == 10

    return lang_x.to(device), vision_x.to(device)


def parse_openflamingo_output(output: str, is_success_description: bool) -> str:
    """Extract the answer from the output combining prompts and answers.

    output example:
    [Observation description]
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table.
    There are white, gray, purple, brown, orange, blue, and pink bowls on the table.
    The white block is in the white bowl. Other bowls are empty. <|endofchunk|>
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table.
    There are white, gray, purple, brown, orange, blue, and pink bowls on the table.
    The white block is in the white bowl. The blue block is in the orange bowl. Other bowls are empty. <|endofchunk|>
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are yellow, orange, pink, red, blue, and cyan blocks on the table.
    There are white, gray, purple, brown, orange, blue, and pink bowls on the table.
    The white block is in the white bowl. The blue block is in the orange bowl. Other bowls are empty. <|endofchunk|>
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are a red block and a green block on the table.
    There are a white bowl and a blue bowl on the table. All the bowls are empty.
    There are a purple zone, a white zone, and a green zone on the table. <|endofchunk|>
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are a red block and a green block on the table.
    There are a white bowl and a blue bowl on the table. All the bowls are empty.
    There are a purple zone, a white zone, and a green zone on the table.
    The green block is in the purple zone. <|endofchunk|>
    <image>Question: What objects are on the table and what's the position of each block?
    Answer: On the current scene, there are a red block and a green block on the table. There are a white bowl and a blue bowl on the table.
    [Success description]
    Please compare the two images and judge whether the robot's action is executed successfully.
    <image><image>The action 'pick up the blue block and place it in the orange bowl' is executed successfully. <|endofchunk|>
    <image><image>The action 'pick up the blue block and place it in the blue bowl' is executed unsuccessfully. <|endofchunk|>
    <image><image>The action 'pick up the green block and place it in the purple zone' is executed successfully. <|endofchunk|>
    <image><image>The action 'pick up the green block and place it in the green zone' is executed unsuccessfully. <|endofchunk|>
    <image><image>The action Pick up the white block and place it on the white bowl. is executed successfully.<|endofchunk|>
    """
    if not is_success_description:
        text = output.split("<|endofchunk|><image>")
        last_example = text[-1]
        answer = last_example.split("Answer:")[1].strip("<|endofchunk|>").strip()
    else:
        text = output.split("<|endofchunk|><image><image>")
        last_example = text[-1]
        answer = last_example.split("Answer:")[1].strip("<|endofchunk|>").strip()

    return answer


def get_description_from_vlm(
        vlm_args,
        example_img_directory,
        test_imgs,
        is_success_description,
        last_instruction=None,
        device=None,
):
    tokenizer, vlm, vis_processor = vlm_args["tokenizer"], vlm_args["vlm"], vlm_args[
        "vis_processor"]

    lang_x, vision_x = get_prompt_for_openflamingo(
        tokenizer,
        vis_processor,
        example_img_directory,
        test_imgs,
        is_success_description,
        last_instruction,
        device
    )
    description = vlm.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=128,
        num_beams=3,
    )
    description = parse_openflamingo_output(
        tokenizer.decode(description[0]),
        is_success_description
    )

    return description
