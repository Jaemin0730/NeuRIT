import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
import re


# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def split_context_and_query(prompt: str):
    """
    - 'Context: ... →' 의 ... 를 context로
    - '→' 이후 전부를 query로 반환
    - context만 제거한 prompt_without_context 도 함께 반환
    """
    m = re.search(r'Context:\s*(.*?)\s*→\s*(.*)', prompt, flags=re.S)
    if not m:
        return "", prompt, prompt  # 패턴 못 찾으면 전체를 query로
    context = m.group(1).strip()
    query   = m.group(2).strip()
    prompt_wo = re.sub(r'Context:\s*.*?\s*→\s*', '', prompt, flags=re.S)  # Context만 제거
    return context, query, prompt_wo

def scaled_input(minimum_activations, maximum_activations, batch_size, num_batch):

    num_points = batch_size * num_batch
    step = (maximum_activations - minimum_activations) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(minimum_activations, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list) # (layer_num, ffn_size)
    max_ig = ig.max()  # maximum attribution score
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def get_context_attr(idx, prompt_without_context, prompt_with_context, answer_obj, args, model, tokenizer, device):
    tokens = tokenizer.tokenize(f" {answer_obj}")  # 항상 공백 추가
    gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])
    
    wo_tokenized_inputs = tokenizer(prompt_without_context, return_tensors="pt")
    wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
    wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)
    
    w_tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
    w_input_ids = w_tokenized_inputs["input_ids"].to(device)
    w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)

    
    # record results
    res_dict = {
        'idx': idx,
        'wo_all_ffn_activations': [],
        'w_all_ffn_activations': [],
        'all_attr_gold': [],
    }

    for tgt_layer in range(model.model.config.num_hidden_layers):
        wo_ffn_activations_dict = dict()
        def wo_forward_hook_fn(module, inp, outp):
            wo_ffn_activations_dict['input'] = inp[0]  # inp type is Tuple

        w_ffn_activations_dict = dict()
        def w_forward_hook_fn(module, inp, outp):
            w_ffn_activations_dict['input'] = inp[0]
        # ========================== get activations when there is no context in the prompt =========================
        wo_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(wo_forward_hook_fn)
        with torch.no_grad():
            wo_outputs = model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
        wo_ffn_activations = wo_ffn_activations_dict['input']
        wo_ffn_activations = wo_ffn_activations[:, -1, :]
        wo_logits = wo_outputs.logits[:, -1, :]
        wo_hook.remove()
        
        # =========================== get activations when there is context in the prompt ============================
        w_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(w_forward_hook_fn)
        with torch.no_grad():
            w_outputs = model(input_ids=w_input_ids, attention_mask=w_attention_mask)
        w_ffn_activations = w_ffn_activations_dict['input']
        w_ffn_activations = w_ffn_activations[:, -1, :]
        w_logits = w_outputs.logits[:, -1, :]
        w_hook.remove()
        
        wo_ffn_activations.requires_grad_(True)
        w_ffn_activations.requires_grad_(True)
        scaled_activations, activations_step = scaled_input(wo_ffn_activations, w_ffn_activations, args.batch_size, args.num_batch)
        scaled_activations.requires_grad_(True)

        # integrated grad at the gold label for each layer
        ig_gold = None
        for batch_idx in range(args.num_batch):
            grad = None
            all_grads = None
            for i in range(0, args.batch_size, args.batch_size_per_inference):
                # print(i, i + args.batch_size_per_inference)
                batch_activations = scaled_activations[i: i + args.batch_size_per_inference] # (batch_size_per_inference, ffn_size)
                batch_w_activations = w_ffn_activations.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, ffn_size)
                batched_w_input_ids = w_input_ids.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                batched_w_attention_mask = w_attention_mask.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)

                def i_forward_hook_change_fn(module, inp):
                    inp0 = inp[0]
                    inp0[:, -1, :] = batch_activations
                    inp = tuple([inp0])
                    return inp
                change_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(i_forward_hook_change_fn)
                
                outputs = model(input_ids=batched_w_input_ids, attention_mask=batched_w_attention_mask)  # (batch, n_vocab), (batch, ffn_size)
                # compute final grad at a layer at the last position
                tgt_logits = outputs.logits[:, -1, :] # (batch, n_vocab)
                tgt_probs = F.softmax(tgt_logits, dim=1) # (batch, n_vocab)
                # grads_i = torch.autograd.grad(torch.unbind(tgt_probs[:, gold_label_id]), batch_activations)
                grads_i = torch.autograd.grad(torch.unbind(tgt_probs[:, gold_label_id]), w_ffn_activations, retain_graph=True) # grads_i[0].shape: (1, ffn_size)
                del tgt_probs

                change_hook.remove()  # check_ffn_activations_dict['output'][:,-1,:]
                
                all_grads = grads_i[0] if all_grads is None else torch.cat((all_grads, grads_i[0]), dim=0)
            grad = all_grads.sum(dim=0)  # (ffn_size)
            ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)
            
        ig_gold = ig_gold * activations_step  # (ffn_size)
        res_dict['wo_all_ffn_activations'].append(wo_ffn_activations.squeeze().tolist())
        res_dict['w_all_ffn_activations'].append(w_ffn_activations.squeeze().tolist())
        res_dict['all_attr_gold'].append(ig_gold.tolist())
        
    return res_dict


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--model_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name as output prefix to indentify each running of experiment.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_id",
                        type=str,
                        default='0',
                        help="available gpu id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # parameters about integrated grad
    parser.add_argument("--num_batch",
                        default=1,
                        type=int,
                        help="Number of examples for each run.")
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="The m in the paper.")
    parser.add_argument("--batch_size_per_inference",
                        default=5,
                        type=int,
                        choices=[1,2,4,5,10,20],
                        help="The batch size for each inference, you can choose an appropriate value from 1, 2, 4, 5, 10, 20 according to the model size and CUDA mamory.")

    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # TODO: To implement multi gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))


    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    output_prefix = f"{args.dataset_name}-{args.model_name}-context"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # prepare dataset
    with open(args.data_path, "r", encoding="utf-8") as fin:
        data = []
        for json_line in fin:
            line = json.loads(json_line)
            data.append(line)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True, low_cpu_mem_usage=True, device_map=device)
    
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    
    # # data parallel
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    model.eval()
    

    tic = time.perf_counter()
    count = 0
    print(f"Start process, dataset size: {len(data)}")
    with jsonlines.open(os.path.join(args.output_dir, output_prefix + '.rlt' + '.jsonl'), 'w') as fw:
        for idx, example in enumerate(tqdm(data)):
            raw_classes = example["classes"]
            if isinstance(raw_classes, list):
                classes = [str(x).strip().strip('.').strip() for x in raw_classes]
            else:
                tmp = str(raw_classes).strip('[]')
                classes = [s.strip().strip("'").strip('.').strip() for s in tmp.split(',') if s.strip()]
            answer_index = example["answer_index"]
            trap_index = 1 if example["answer_index"]==0 else 0 
            answer_obj = classes[answer_index]
            trap_obj= classes[trap_index]
            prompt_with_context = example["prompt"]
            ctx_text, query_text, prompt_without_context = split_context_and_query(prompt_with_context)
            
            res_dict = get_context_attr(idx, prompt_without_context, prompt_with_context, answer_obj, args, model, tokenizer, device)
            res_dict["all_attr_gold"] = convert_to_triplet_ig(res_dict["all_attr_gold"])
            
            fw.write(res_dict)
            count += 1
        
    print(f"Saved in {os.path.join(args.output_dir, output_prefix + '.rlt' + '.jsonl')}")

    toc = time.perf_counter()
    time_str = f"***** Costing time: {toc - tic:0.4f} seconds *****"
    print(time_str)
    json.dump(time_str, open(os.path.join(args.output_dir, output_prefix + '.time.json'), 'w'))


if __name__ == "__main__":
    main()