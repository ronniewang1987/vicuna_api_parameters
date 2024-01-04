---
library_name: peft
base_model: lmsys/vicuna-7b-v1.5
---

# 基于vicuna微调的接口参数自动生成模型

接口参数自动生成（Interface parameters are automatically generated）

对应的huggingface仓库：
https://huggingface.co/ronniewy/vicuna_api_parameters

## How to use（如何使用）

### 方式一
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

config = PeftConfig.from_pretrained("ronniewy/vicuna_api_parameters")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
model = PeftModel.from_pretrained(model, "ronniewy/vicuna_api_parameters")

...
```

### 方式二（推荐）
使用本仓库的启动文件`cli.py`（依赖fschat项目"pip install fschat"）：
```shell
python cli.py --model-path ronniewy/vicuna_api_parameters
```

命令会启动对话模式：
```text
(myenv) [root@VM-230-191-tencentos vicuna_api_parameters]# python cli.py 
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.33s/it]

human: 
```

**在human后的光标后输入json schema定义的api，会自动生成合法/非法的参数供使用。**

example：
比如现有json schema定义的api：

{"type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "genre": {"type": "string"}, "published": {"type": "boolean"}, "rating": {"type": "number", "minimum": 0, "maximum": 10}, "tags": {"type": "array", "items": {"type": "string"}}}}

```text
(myenv) [root@VM-230-191-tencentos vicuna_api_parameters]# python cli.py 
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.33s/it]

human:  {"type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "genre": {"type": "string"}, "published": {"type": "boolean"}, "rating": {"type": "number", "minimum": 0, "maximum": 10}, "tags": {"type": "array", "items": {"type": "string"}}}}
gpt: {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Classic", "published": true, "rating": 9.0, "tags": ["American literature", "Roaring Twenties"]}

human: 
```

## Supplement（补充）

### json schema定义api

参考官网：[json schema](https://json-schema.org/)

### 合法/非法校验
参考如下python代码：
```python
from jsonschema import Draft7Validator


# JSON Schema定义
schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer', 'minimum': 18},
        'email': {'type': 'string', 'format': 'email'},
        'address': {'type': 'string'},
        'phone': {'type': 'string', 'pattern': '\\d{3}-\\d{3}-\\d{4}'}
    }
}

# 待验证的JSON数据
data = {"name": "user123", "age": 15, "email": "xx@qq.com", "address": "xxxx", "phone": "123"}

validator = Draft7Validator(schema)

# 进行校验
errors = validator.iter_errors(data)

# 输出验证错误信息
"""
输出（有多个错误会输出多个）：
Validation error: 15 is less than the minimum of 18
Validation error: '123' does not match '\\d{3}-\\d{3}-\\d{4}'
"""
error_count = 0
for error in errors:
    print(f"Validation error: {error.message}")
    error_count += 1

# 如果没有错误，打印"合法"
if error_count == 0:
    print("合法")
```

### 构造请求

有了模型生成的参数，有了自动校验参数的合法性，就能构造请求对接口测试并验证是否符合预期。
