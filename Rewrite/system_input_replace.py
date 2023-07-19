import json
lines = open('/Users/jinqigong/Desktop/evals/evals/registry/data/chess_piece_count/fuzzy_match.jsonl').readlines()
data_list = [json.loads(line) for line in lines]
print(data_list[0])
for d in data_list:
    d['input'][0]['content'] = "You are ChessGPT. A helpful AI chatbot that understands chess fundamental and helps people analyse their moves. Follow the instructions given to the point."
print(data_list[0])
new_lines = [json.dumps(d) for d in data_list]
open('/Users/jinqigong/Desktop/evals/evals/registry/data/chess_piece_count/fuzzy_match.jsonl', 'w') .write('\n'.join(new_lines))