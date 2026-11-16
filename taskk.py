from llm_sdk import Small_LLM_Model
from pathlib import Path
import numpy as np
import json


def load_json(path, key):
  with open(path, "r") as f:
    [d.get(key) for d in json.load(f)]

def token_tree(model, names):
  root = {}
  for n in names:
    node = root
    for tid in model.encode(n):
      node = node.setdefault(tid, {})
    node["end"] = n
  return root

def context_prompt(prompt, names):
  return (
    "your are a function calling assitant\n"
    f"here is the funtion definition: {prompt}\n"
    "your job is to pick the correct funtion name\n"
    f"here is the list you should be picking from: {names}\n"
    "Answer: "
  )

def build_dict(model, trie, prompt, names):
  input_ids = model.encode(context_prompt(prompt, names)).tolist()[0]
  node = trie
  while True:
    logits = np.array(model.get_logit_from_input_ids())
    mask = np.full_like(logits, -np.inf)
    valid_ids = [tid for tid in node if tid != "end"]
    mask[valid_ids] = logits[valid_ids]
    pred_token = np.argmax(mask)

    node = node[pred_token]
    if "end" in node:
      break
    input_ids.append(pred_token)

  return {
    "prompt" : prompt,
    "name" : node["end"]
  }


def main():
  names = load_json("data/input/funtions_definition", "name")
  prompts = load_json("data/input/function_calling_test", "prompt")
  model = Small_LLM_Model()

  trie = token_tree(model, names)
  result = [build_dict(model, trie, p, names) for p in prompts]

  out_path = Path("data/output/output.json")
  out_path.parent.mkdir(exist_ok = True)

  with open(out_path, "w") as f:
    json.dump(result, f)


if __name__ == "__main__":
  main()
