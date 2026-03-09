import json
import random

def create_eval_files(input_file, sample_size=20):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_keys = list(data['text'].keys())
        actual_size = min(sample_size, len(all_keys))
        sampled_keys = random.sample(all_keys, actual_size)

        ground_truth = {"text": {}, "sentiment": {}}
        test_input = {"text": {}, "sentiment": {}}

        for i, key in enumerate(sampled_keys):
            text_val = data["text"][key]
            sent_val = data["sentiment"][key]
            
            ground_truth["text"][str(i)] = text_val
            ground_truth["sentiment"][str(i)] = sent_val
            
            test_input["text"][str(i)] = text_val
            test_input["sentiment"][str(i)] = ""

        with open('random_with_sentiment.json', 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, ensure_ascii=False, indent=4)
            
        with open('random_without_sentiment.json', 'w', encoding='utf-8') as f:
            json.dump(test_input, f, ensure_ascii=False, indent=4)

        print(f"✅ Created 'random_with_sentiment.json' (with results)")
        print(f"✅ Created 'random_without_sentiment.json' (sentiment is empty)")
        print(f"Total samples: {actual_size}")

    except Exception as e:
        print(f"❌ Error: {e}")

create_eval_files('train_sentiment.json', sample_size=20)