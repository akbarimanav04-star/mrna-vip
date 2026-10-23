"""Run the whole project in order:  python run_pipeline.py"""
from src import attention, dashboard, data_pipeline, eda, rag_eval, rag_index, report_generator, risk_model
from src.config import CLEAN_CSV, FIG_DIR


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Week 1: data + classifier ===")
    df = data_pipeline.run()
    eda.make_figures()
    risk_model.train_and_evaluate(data_pipeline.load_clean(CLEAN_CSV))
    print("\n=== Week 2: self-attention ===")
    attention.run()
    print("\n=== Week 3: RAG index ===")
    rag_index.build_index()
    try:
        rag_eval.run()                       # optional chunk-size experiment (fills a table in the Guideline tab)
    except Exception as exc:                 # never let an optional step block Week 4
        print(f"WARNING: optional retrieval experiment skipped ({type(exc).__name__}: {exc})")
    print("\n=== Week 4: reports + dashboard ===")
    report_generator.run()
    dashboard.build_dashboard()
    print("\nDone. See outputs/.")


if __name__ == "__main__":
    main()
