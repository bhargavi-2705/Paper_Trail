"""CLI quiz over topics the corpus covers, prioritising your weak spots.

    python scripts/run_quiz.py --topic attention
    python scripts/run_quiz.py --adaptive          # uses coverage.json + quiz history
    python scripts/run_quiz.py --validate-judge    # re-grade saved answers by hand
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                 # noqa: E402
from src.judge import validate_judge               # noqa: E402
from src.llm import get_llm                        # noqa: E402
from src.quiz import generate_quiz, grade, next_topics  # noqa: E402
from src.retrieve import Retriever                 # noqa: E402

HISTORY = ROOT / "eval" / "results" / "quiz_history.json"


def load_history():
    return json.loads(HISTORY.read_text()) if HISTORY.exists() else []


def save_history(h):
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(h, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--validate-judge", action="store_true")
    args = ap.parse_args()

    history = load_history()

    if args.validate_judge:
        graded = [h for h in history if h.get("score") is not None]
        todo = [h for h in graded if h.get("human_label") is None]
        print(f"{len(todo)} answers to hand-grade (1 = I'd call it correct, 0 = not).")
        for h in todo:
            print("=" * 70)
            print(f"Q: {h['question']}\nYour answer: {h['student_answer']}\n"
                  f"Model answer: {h['model_answer']}\nJudge: {h['score']} - {h['feedback']}")
            v = input("your label [0/1, enter=skip]: ").strip()
            if v in ("0", "1"):
                h["human_label"] = int(v)
        save_history(history)
        stats = validate_judge([{"judge_score": h.get("score"), "human_label": h.get("human_label")}
                                for h in history])
        print(f"\njudge vs me: n={stats['n']} agreement={stats['agreement']} kappa={stats['kappa']}")
        print("Report these next to any quiz accuracy number.")
        return

    cfg = load_config()
    retriever, llm = Retriever(cfg), get_llm(cfg)

    if args.adaptive:
        cov_path = ROOT / "eval" / "results" / "coverage.json"
        if not cov_path.exists():
            sys.exit("Run scripts/run_coverage.py first.")
        topics = next_topics(json.loads(cov_path.read_text())["topics"], history, limit=2)
    elif args.topic:
        topics = [args.topic]
    else:
        sys.exit("Pass --topic or --adaptive.")
    print(f"topics: {', '.join(topics)}\n")

    for topic in topics:
        for item in generate_quiz(topic, retriever, llm, n=args.n):
            print("=" * 70)
            print(f"[{topic}] {item['question']}")
            student = input("your answer (enter to skip): ").strip()
            if not student:
                continue
            g = grade(item, student, llm)
            print(f"  -> {'correct' if g.get('correct') else 'not quite'} "
                  f"(score {g.get('score')})\n  {g.get('feedback')}")
            print(f"  source: {item['source']['doc']} p.{item['source']['page']}")
            history.append({**{k: item[k] for k in ("topic", "question", "model_answer")},
                            "student_answer": student, "score": g.get("score"),
                            "correct": g.get("correct"), "feedback": g.get("feedback"),
                            "human_label": None})
    save_history(history)
    print(f"\nsaved {len(history)} graded answers. Run --validate-judge to check the grader.")


if __name__ == "__main__":
    main()
