import numpy
import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

def readjson(path, key):
    lst = []
    with open(path, "r") as f:
        pr = json.load(f)


    return [p.get(key) for p in pr]

def valid_logits(functions, res):
    l = len(res)
    v_token = set()
    for f in functions:
        if len(f) <= l:
            continue
        if f[:l] == res:
            next_char = f[l]
            token_ids = model.encode(next_char).tolist()[0]
            if len(token_ids) == 1:
                v_token.add(token_ids[0])
    return list(v_token)


def mask(logits, functions, res):
    logits_copy = numpy.array(logits)
    masked = numpy.full_like(logits_copy, -numpy.inf)
    v_ids = valid_logits(functions, res)
    if not v_ids:
        return None
    masked[v_ids] = logits_copy[v_ids]
    return masked


if __name__ == "__main__":
    prompts = readjson("data/input/function_calling_tests.json", 'prompt')
    functions = readjson("data/input/functions_definition.json", 'name')

    context = (
        "Pick the exact function name that matches the question, for example:\n"
        "Question: What is the square root of 16?\n"
        "Function: fn_get_square_root\n"
        "Question: Greet shrek\n"
        "Function: fn_greet\n"
        "Question: Reverse the string 'hello'\n"
        "Function: fn_reverse_string\n"
    )

    for p in prompts:
        res = ''
        print(p)
        while True:
            if res in functions:
                break
            ids = model.encode(f"{context}{p}{res}").tolist()[0]
            logits = model.get_logits_from_input_ids(ids)
            masked = mask(logits, functions, res)
            if masked is None:
                print('out')
                break
            token_id = int(numpy.argmax(masked))
            token_str = model.decode([token_id])
            res += token_str
        print('res:', res)
        print('-' * 30)
