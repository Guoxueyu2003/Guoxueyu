import os
import json
import torch
import argparse
import transformers
from PIL import Image

processor = transformers.AutoProcessor.from_pretrained("./layoutlmv3-large", apply_ocr=False)
spatial_position_id = 150000
img_patch_id = 150001

# model args
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='model directory')
parser.add_argument('--img_dir', type=str, help='image directory')
parser.add_argument('--ocr_dir', type=str, help='ocr directory')
parser.add_argument('--instruction', type=str, help='question of the image')
args = parser.parse_args()

def normalize_bbox(bbox, src_size, dst_size):

    src_w, src_h = src_size["width"], src_size["height"]
    dst_w, dst_h = dst_size["width"], dst_size["height"]
    x1, y1, x2, y2 = bbox
    # 进行坐标排序
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    # 线性归一化映射
    x1 = int(x_min / src_w * dst_w)
    y1 = int(y_min / src_h * dst_h)
    x2 = int(x_max / src_w * dst_w)
    y2 = int(y_max / src_h * dst_h)
    # 进行坐标限制
    x1 = max(0, min(x1, dst_w))
    y1 = max(0, min(y1, dst_h))
    x2 = max(0, min(x2, dst_w))
    y2 = max(0, min(y2, dst_h))

    return [x1, y1, x2, y2]


def main():
    # 加载配置
    config = transformers.AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    generator_config = transformers.GenerationConfig.from_pretrained(
        args.model_dir
    ).to_dict()

    # 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        trust_remote_code=True,
    )
    model = model.eval()
    model = model.to(torch.float32)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)

    # 加载图像
    img_dir = args.img_dir
    image = Image.open(img_dir).convert('RGB')
    width, height = image.size
    image = processor(image,
                      [''],
                      boxes=[[0,0,0,0],],
                      return_tensors="pt",
                      padding=True)['pixel_values'][0]

    # 加载OCR数据
    ocr_dir = args.ocr_dir
    ocr_data = json.load(open(ocr_dir))

    prompt = args.instruction
    fore_prompt = '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>'
    fore_prompt += '<|start_header_id|>user<|end_header_id|>\n\nGiving the document image patches,'
    fore_llm_value_ids = tokenizer.encode(fore_prompt, add_special_tokens=False,)
    fore_llm_value_ids = [tokenizer.bos_token_id,] + fore_llm_value_ids
    fore_llm_value_ids = fore_llm_value_ids + [img_patch_id,] * 196
    fore_prompt = ', and text content and its location in form of "text, [left, top, right, bottom]":\n'
    fore_llm_value_ids = fore_llm_value_ids + tokenizer.encode(fore_prompt, add_special_tokens=False,)
    aft_prompt = "\n" + prompt +  '<|eot_id|>'
    aft_prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
    aft_llm_value_ids = tokenizer.encode(aft_prompt, add_special_tokens=False,)

    #OCR文本处理
    bbox = []
    for _, line in enumerate(ocr_data):
        # 移除OCR文本首尾的空白字符
        line_text = line['text'].strip()
        # 将文本转换成为Token序列并添加自定义的空间位置标识符并保留换行信息
        tokenized = tokenizer.encode(line_text, add_special_tokens=False) + \
            [spatial_position_id, ] + \
                tokenizer.encode("\n", add_special_tokens=False)
        #将每行OCR文本的序列拼接
        fore_llm_value_ids += tokenized
        # 边界框归一化
        line_box = line['box']
        #坐标归一化
        norm_box = normalize_bbox(line_box, {"width": width, "height": height},
                                  {"width": 1000, "height": 1000})
        bbox += [norm_box,]

    # 模型输入
    input_ids = fore_llm_value_ids + aft_llm_value_ids
    position_ids = list(range(len(input_ids)))

    input_ids = torch.LongTensor(input_ids)
    position_ids = torch.LongTensor(position_ids)
    bbox = torch.LongTensor(bbox)

    input = {
        "input_ids": input_ids.unsqueeze(0).to(model.device),
        "position_ids": position_ids.unsqueeze(0).to(model.device),
        "bbox": bbox.unsqueeze(0).to(model.device),
        "pixel_values": image.unsqueeze(0).to(model.device),
    }

    output = model.generate(**input, **generator_config).cpu()
    response = tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)
    print(f"Response: {response}")

if __name__ == '__main__':
    main()
