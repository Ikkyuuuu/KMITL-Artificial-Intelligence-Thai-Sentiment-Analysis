import json
from sklearn.metrics import accuracy_score, classification_report

def check_model_results(prediction_file, random_with_sentiment):
    try:
        with open(prediction_file, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        with open(random_with_sentiment, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)

        y_true = []
        y_pred = []
        
        for key in truth_data["sentiment"]:
            y_true.append(truth_data["sentiment"][key])
            y_pred.append(pred_data["sentiment"].get(key, "MISSING"))

        print("="*50)
        print("📊 MODEL PERFORMANCE REPORT")
        print("="*50)
        print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.4f}")
        print("\nDetailed Metrics:")
        print(classification_report(y_true, y_pred))
        
        print("="*50)
        print("🔍 MISMATCHED SAMPLES")
        print("="*50)
        mismatch_count = 0
        for key in truth_data["sentiment"]:
            true_label = truth_data["sentiment"][key]
            pred_label = pred_data["sentiment"].get(key)
            
            if true_label != pred_label:
                mismatch_count += 1
                text_snippet = truth_data["text"][key][:80] + "..."
                print(f"ID {key}:")
                print(f"  - Actual: {true_label}")
                print(f"  - Model : {pred_label}")
                print(f"  - Text  : {text_snippet}\n")
        
        if mismatch_count == 0:
            print("Perfect match! No errors found.")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")

check_model_results('sentiment_result.json', 'random_with_sentiment.json')